# Signed Index and Negative Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public `Index`/`Axes`/`Shape` signed (`std::ptrdiff_t`) so that axes and element indices accept numpy-style negative values, while all internal computation keeps using `size_t`.

**Architecture:** `index.hpp` becomes a header-only class template `BasicIndex<T>` with `Index = BasicIndex<std::ptrdiff_t>` (public) and `detail::UIndex = BasicIndex<size_t>` (internal). Public functions normalize their arguments once at the entry point with `normalize_*` / `to_internal_shape`, which throw `std::out_of_range` / `std::invalid_argument`; everything after that works on `detail::UIndex`. Internal code calls `detail::transpose_impl` / `detail::reshape_impl` and a tagged constructor `Tensor(comm, UIndex, urank, detail::internal)` to avoid round trips through the public type.

**Tech Stack:** C++17, CMake ≥ 3.16, GoogleTest (via the infrastructure from PR #10), MPI + ScaLAPACK / LAPACK.

**Spec:** `docs/superpowers/specs/2026-10-02-signed-index-design.md`

## Global Constraints

- Public element type: `std::ptrdiff_t`; internal: `size_t`. `Axes`, `Shape`, `Index` stay aliases of the same class.
- The single-argument constructor stays implicit (`Axes a = 2;` compiles); `Index(3)` and `Index{3}` are both `[3]`.
- Out-of-range axis or element index → `std::out_of_range`; invalid shape or length mismatch → `std::invalid_argument`. Messages include the value, its position, and the rank or size.
- Normalization happens before any MPI communication or modification of a tensor.
- Slices: raw `begin == end` in the `Index` versions means the full axis; otherwise `begin` ∈ `[-n, n)`, `end` ∈ `[-n, n]`, and `begin >= end` after normalization throws `std::out_of_range`. Scalar versions do not have the raw-equal rule.
- Count arguments (`upper_rank`, `urank`, `target_rank`, `oversamp`) stay `size_t`.
- Existing tests in `tests/tensor` and `tests/file_io` must pass **without modification**; `tests/legacy` is not modified and must pass with `-DBUILD_LEGACY_TESTS=ON`.
- New and modified code produces no warnings with `-Wall -Wextra -Wsign-compare`.
- Commit messages: plain imperative sentence, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Work on branch `feature/signed-index`; never push without asking.

## Review Focus

1. **Brace-initialized shapes from `size_t` variables** (`Shape{n, n + 1}` with `size_t n`): the new `initializer_list<std::ptrdiff_t>` constructor makes this a narrowing error at compile time. Expected: existing code in the repository still compiles. → Task 4 builds `examples/` and `tests/legacy`; any such site is reported, not silently cast.
2. **`range()` with negative values used as axes** (`transpose(A, range(-4, 0))`): `range` now produces signed values, and they must be normalized like any other `Axes`. Expected: identical to `range(0, 4)`. → `NegativeIndex.TransposeFreeAndMember` (Task 3) checks it. (A negative value that lands on a *duplicate* axis, e.g. `Axes(0, -4)` for rank 4, is out of scope per the spec and is left to the existing Debug assertions.)
3. **Exceptions in MPI runs with 4 processes**: every rank must throw at the same point and continue. → `NegativeIndexExceptions.CollectiveContinuesAfterThrow` (Task 3) runs a `tensordot` after catching.
4. **Tensor left unchanged after a throw** (member `transpose`, `set_slice`, `multiply_vector`): Expected: no partial modification. → `NegativeIndexExceptions.TensorUnchangedAfterThrow` (Task 3).
5. **`shape()` returning by value**: `const Shape& s = a.shape();` must still work (lifetime extension), and `a.shape()[k]` must still be usable in arithmetic with `size_t`. → existing tests use both patterns and must pass unchanged (Task 3, Step 13).

## Local environment notes

- Intel MPI + icpx locally (1 core); build dirs under `build/` (git-ignored):
  - MPI: `cmake -S . -B build/si-mpi -G Ninja -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug`
  - No MPI: `cmake -S . -B build/si-nompi -G Ninja -DBUILD_TESTS=ON -DENABLE_MPI=OFF -DCMAKE_BUILD_TYPE=Debug`
- Full test command used at the end of every task:
  `ctest --test-dir build/si-mpi --output-on-failure && ctest --test-dir build/si-nompi --output-on-failure`

## File Structure

| Path | Change |
|---|---|
| `include/mptensor/index.hpp` | Rewrite: `BasicIndex<T>`, aliases, `range`, conversion functions, `detail` helpers |
| `include/mptensor/index_constructor.hpp`, `index_constructor.py`, `src/index.cc` | Delete |
| `src/CMakeLists.txt`, `src/Makefile.depend`, `doc/doxygen/Doxyfile`, `doc/doxygen/Doxyfile.in` | Remove references to the deleted files |
| `include/mptensor/tensor.hpp` | Declarations: internal types, tagged constructor, `internal_shape()`, `ptrdiff_t` scalar arguments, `detail::transpose_impl/reshape_impl` |
| `include/mptensor/tensor_impl.hpp` | Normalize at entry points; internal code on `detail::UIndex` |
| `include/mptensor/rsvd_impl.hpp`, `include/mptensor/file_io/load.hpp`, `src/tensor.cc` | Internal code on `detail::UIndex` |
| `tests/index/CMakeLists.txt`, `tests/index/basic_index.cc`, `tests/index/conversions.cc` | New unit tests |
| `tests/tensor/negative_index.cc` | New tensor-level tests |
| `tests/CMakeLists.txt`, `tests/tensor/CMakeLists.txt` | Register the new tests |

---

### Task 1: `BasicIndex<T>` template (no behavior change yet)

`Index` stays `BasicIndex<size_t>` in this task, so the library is unchanged; the generated constructors are replaced and `tests/index` is introduced.

**Files:**
- Rewrite: `include/mptensor/index.hpp`
- Delete: `include/mptensor/index_constructor.hpp`, `include/mptensor/index_constructor.py`, `src/index.cc`
- Modify: `src/CMakeLists.txt:21`, `src/Makefile.depend`, `doc/doxygen/Doxyfile:1049`, `doc/doxygen/Doxyfile.in:1049`, `tests/CMakeLists.txt`
- Create: `tests/index/CMakeLists.txt`, `tests/index/basic_index.cc`

**Interfaces:**
- Produces:
  - `template <typename T> class mptensor::BasicIndex` with `value_type`, `index_t`, constructors `()`, `(const index_t&)`, `(std::initializer_list<T>)`, variadic `(Ints...)`; members `operator[]`, `size`, `push(T)`, `resize`, `assign(size_t, const T[])`, `sort`, `inverse() const`, `operator==`, `operator+=`; free `operator<<`, `operator+`.
  - `template <typename T, typename I> T mptensor::detail::checked_index_cast(I)` (throws `std::out_of_range`).
  - `using Index = BasicIndex<std::size_t>;` (temporary, flipped in Task 3), `namespace detail { using UIndex = BasicIndex<std::size_t>; }`.
  - `Index range(size_t start, size_t stop)`, `Index range(size_t stop)` (unchanged semantics).

- [ ] **Step 1: Write the failing test `tests/index/basic_index.cc`**

```cpp
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <mptensor/index.hpp>

namespace {

using SIndex = mptensor::BasicIndex<std::ptrdiff_t>;
using UIndex = mptensor::BasicIndex<std::size_t>;

TEST(BasicIndex, VariadicAcceptsMixedIntegerTypes) {
  const int a = 1;
  const long b = -2;
  const std::size_t c = 3;
  const std::ptrdiff_t d = -4;
  const SIndex idx(a, b, c, d);
  ASSERT_EQ(idx.size(), 4u);
  EXPECT_EQ(idx[0], 1);
  EXPECT_EQ(idx[1], -2);
  EXPECT_EQ(idx[2], 3);
  EXPECT_EQ(idx[3], -4);
}

TEST(BasicIndex, ParenthesesAndBracesAgree) {
  EXPECT_EQ(SIndex(0, 1, -1), (SIndex{0, 1, -1}));
  EXPECT_EQ(UIndex(0, 1, 2), (UIndex{0, 1, 2}));
}

TEST(BasicIndex, SingleArgumentIsLengthOne) {
  const SIndex a(3);
  const SIndex b{3};
  ASSERT_EQ(a.size(), 1u);
  EXPECT_EQ(a[0], 3);
  EXPECT_EQ(a, b);
}

TEST(BasicIndex, ImplicitConversionFromInteger) {
  const SIndex a = 2;
  ASSERT_EQ(a.size(), 1u);
  EXPECT_EQ(a[0], 2);
}

TEST(BasicIndex, MoreThanSixteenArguments) {
  const SIndex idx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                   17);
  ASSERT_EQ(idx.size(), 18u);
  EXPECT_EQ(idx[17], 17);
}

TEST(BasicIndex, NegativeValueIntoUnsignedThrows) {
  EXPECT_THROW(UIndex(0, -1), std::out_of_range);
}

TEST(BasicIndex, HugeUnsignedValueIntoSignedThrows) {
  const std::size_t huge = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(SIndex(huge), std::out_of_range);
}

TEST(BasicIndex, FromVector) {
  const std::vector<std::ptrdiff_t> v{5, -6};
  const SIndex idx(v);
  ASSERT_EQ(idx.size(), 2u);
  EXPECT_EQ(idx[1], -6);
}

TEST(BasicIndex, PushResizeAssign) {
  UIndex idx;
  idx.push(4);
  idx.push(7);
  EXPECT_EQ(idx, UIndex(4, 7));
  idx.resize(3);
  EXPECT_EQ(idx, UIndex(4, 7, 0));
  const std::size_t raw[] = {9, 8};
  idx.assign(2, raw);
  EXPECT_EQ(idx, UIndex(9, 8));
}

TEST(BasicIndex, SortInverseConcatenate) {
  UIndex p(2, 0, 1);
  EXPECT_EQ(p.inverse(), UIndex(1, 2, 0));
  p.sort();
  EXPECT_EQ(p, UIndex(0, 1, 2));
  EXPECT_EQ(UIndex(0, 1) + UIndex(2, 3), UIndex(0, 1, 2, 3));
  SIndex s(-1, 0);
  s += SIndex(5);
  EXPECT_EQ(s, SIndex(-1, 0, 5));
}

TEST(BasicIndex, StreamFormat) {
  std::ostringstream os;
  os << SIndex(0, -1, 2) << " " << UIndex();
  EXPECT_EQ(os.str(), "[0, -1, 2] []");
}

TEST(BasicIndex, Range) {
  EXPECT_EQ(mptensor::range(3), mptensor::Index(0, 1, 2));
  EXPECT_EQ(mptensor::range(2, 4), mptensor::Index(2, 3));
  EXPECT_EQ(mptensor::range(2, 2).size(), 0u);
}

}  // namespace
```

- [ ] **Step 2: Register `tests/index`**

Create `tests/index/CMakeLists.txt`:

```cmake
add_executable(test_index basic_index.cc)
target_link_libraries(test_index PRIVATE mptensor_gtest_main)

mptensor_add_test(index/test_index test_index)
```

In `tests/CMakeLists.txt`, before `add_subdirectory(tensor)`, add `add_subdirectory(index)`.

- [ ] **Step 3: Run it to verify it fails**

Run: `cmake -S . -B build/si-mpi -G Ninja -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug > /dev/null && cmake --build build/si-mpi --target test_index 2>&1 | grep -m3 error`
Expected: compile errors such as `no template named 'BasicIndex' in namespace 'mptensor'`.

- [ ] **Step 4: Rewrite `include/mptensor/index.hpp`**

Keep the existing license header and the Doxygen `\file` block (update `\brief` to "header file of BasicIndex class template"), then:

```cpp
#ifndef _INDEX_HPP_
#define _INDEX_HPP_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mptensor {
//! \ingroup Index
//! \{

namespace detail {
//! Convert an integer to \c T, or throw std::out_of_range if it does not fit.
template <typename T, typename I>
T checked_index_cast(I value) {
  static_assert(std::is_integral_v<I>, "index values must be integers");
  bool fits = true;
  if constexpr (std::is_signed_v<I> && std::is_unsigned_v<T>) {
    fits = (value >= 0);
  } else if constexpr (std::is_unsigned_v<I> && std::is_signed_v<T>) {
    fits = (value <= static_cast<std::make_unsigned_t<T>>(
                         std::numeric_limits<T>::max()));
  }
  if (!fits) {
    std::ostringstream ss;
    ss << "mptensor: index value " << +value << " is out of range for "
       << (std::is_signed_v<T> ? "a signed" : "an unsigned") << " index";
    throw std::out_of_range(ss.str());
  }
  return static_cast<T>(value);
}
}  // namespace detail

//! List of non-negative or signed integers used as an index, axes, or shape.
/*!
  \c Index (= \c Axes = \c Shape) is the public type. \c detail::UIndex is
  used inside the library.
*/
template <typename T>
class BasicIndex {
 public:
  using value_type = T;
  using index_t = std::vector<T>;

  BasicIndex() = default;
  BasicIndex(const index_t& index) : idx(index) {}
  BasicIndex(std::initializer_list<T> list) : idx(list) {}

  //! Python-like list literal, e.g. <tt>Index(0, 1, -1)</tt>.
  /*! Not explicit: <tt>Axes a = 2;</tt> creates <tt>[2]</tt>. */
  template <typename... Ints,
            typename = std::enable_if_t<(sizeof...(Ints) > 0) &&
                                        (std::is_integral_v<Ints> && ...)>>
  BasicIndex(Ints... js) : idx{detail::checked_index_cast<T>(js)...} {}

  const T& operator[](size_t i) const { return idx[i]; }
  T& operator[](size_t i) { return idx[i]; }
  size_t size() const { return idx.size(); }
  void push(T i) { idx.push_back(i); }
  void resize(size_t n) { idx.resize(n); }
  void assign(size_t n, const T j[]) { idx.assign(j, j + n); }
  void sort() { std::sort(idx.begin(), idx.end()); }

  //! Inverse permutation: <tt>inv[(*this)[i]] = i</tt>.
  BasicIndex inverse() const {
    BasicIndex inv;
    inv.resize(size());
    for (size_t i = 0; i < size(); ++i) {
      inv[static_cast<size_t>(idx[i])] = static_cast<T>(i);
    }
    return inv;
  }

  bool operator==(const BasicIndex& rhs) const { return idx == rhs.idx; }

  BasicIndex& operator+=(const BasicIndex& rhs) {
    idx.insert(idx.end(), rhs.idx.begin(), rhs.idx.end());
    return *this;
  }

 private:
  index_t idx;
};

/*! The format is the same as a list of python, for example "[0, 1, 2]". */
template <typename T>
std::ostream& operator<<(std::ostream& os, const BasicIndex<T>& idx) {
  os << "[";
  if (idx.size() > 0) os << idx[0];
  for (size_t i = 1; i < idx.size(); ++i) os << ", " << idx[i];
  os << "]";
  return os;
}

//! Joint two indices: Index(0,1) + Index(2,3) = Index(0,1,2,3)
template <typename T>
BasicIndex<T> operator+(const BasicIndex<T>& lhs, const BasicIndex<T>& rhs) {
  return (BasicIndex<T>(lhs) += rhs);
}

using Index = BasicIndex<std::size_t>;

namespace detail {
using UIndex = BasicIndex<std::size_t>;
}  // namespace detail

//! Create an increasing sequence. It is similar to range() in python.
inline Index range(const size_t start, const size_t stop) {
  assert(start <= stop);
  Index index;
  index.resize(stop - start);
  for (size_t i = start; i < stop; ++i) index[i - start] = i;
  return index;
}
inline Index range(const size_t stop) { return range(0, stop); }

//! \}
}  // namespace mptensor

#endif  //  _INDEX_HPP_
```

- [ ] **Step 5: Delete the generated constructors and `index.cc`; remove their references**

```bash
git rm include/mptensor/index_constructor.hpp include/mptensor/index_constructor.py src/index.cc
sed -i '/PATTERN "index_constructor.py" EXCLUDE/d' src/CMakeLists.txt
sed -i 's|^EXCLUDE                = .*index_constructor.py$|EXCLUDE                =|' doc/doxygen/Doxyfile doc/doxygen/Doxyfile.in
make -C src depend
grep -n "index_constructor\|index\.o\|index\.cc" src/Makefile.depend src/CMakeLists.txt doc/doxygen/Doxyfile doc/doxygen/Doxyfile.in
```
Expected: the last `grep` prints nothing.

- [ ] **Step 6: Run the new tests and the whole suite**

Run:
```bash
cmake -S . -B build/si-mpi > /dev/null && cmake --build build/si-mpi -j2 2>&1 | grep -E " error|error:" ; ctest --test-dir build/si-mpi --output-on-failure
cmake -S . -B build/si-nompi -G Ninja -DBUILD_TESTS=ON -DENABLE_MPI=OFF -DCMAKE_BUILD_TYPE=Debug > /dev/null && cmake --build build/si-nompi -j2 2>&1 | grep -E " error|error:" ; ctest --test-dir build/si-nompi --output-on-failure
```
Expected: no compile errors; MPI: 17 tests pass (4 `index/` + 4 `tensor/` + 9 `file_io/`); no MPI: 5 tests pass.

- [ ] **Step 7: Commit**

```bash
git add include/mptensor/index.hpp src/CMakeLists.txt src/Makefile.depend doc/doxygen/Doxyfile doc/doxygen/Doxyfile.in tests/CMakeLists.txt tests/index
git commit -m "Replace generated Index constructors with a BasicIndex class template

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Conversion functions

**Files:**
- Modify: `include/mptensor/index.hpp` (append before the closing `//! \}`)
- Create: `tests/index/conversions.cc`
- Modify: `tests/index/CMakeLists.txt`

**Interfaces:**
- Consumes: `BasicIndex<T>`, `detail::UIndex` (Task 1).
- Produces (namespace `mptensor`):
  - `size_t normalize_axis(std::ptrdiff_t a, size_t rank)`
  - `detail::UIndex normalize_axes(const BasicIndex<std::ptrdiff_t>& axes, size_t rank)`
  - `size_t normalize_index(std::ptrdiff_t i, size_t n)`
  - `detail::UIndex normalize_index(const BasicIndex<std::ptrdiff_t>& idx, const detail::UIndex& shape)`
  - `size_t normalize_slice_end(std::ptrdiff_t e, size_t n)`
  - `detail::UIndex to_internal_shape(const BasicIndex<std::ptrdiff_t>& s)`
  - `BasicIndex<std::ptrdiff_t> to_public(const detail::UIndex& u)`
  - `detail::UIndex detail::identity_axes(size_t n)` — `[0, 1, ..., n-1]`
  - `std::pair<size_t, size_t> detail::normalize_slice_range(std::ptrdiff_t begin, std::ptrdiff_t end, size_t n, size_t axis)` — scalar slice; throws if empty
  - `void detail::normalize_slice_ranges(const BasicIndex<std::ptrdiff_t>& begin, const BasicIndex<std::ptrdiff_t>& end, const detail::UIndex& shape, detail::UIndex& ubegin, detail::UIndex& uend)` — `Index` slice; raw `begin[r] == end[r]` gives `[0, shape[r])`

(After Task 3, `BasicIndex<std::ptrdiff_t>` is `Index`/`Axes`/`Shape`.)

- [ ] **Step 1: Write the failing test `tests/index/conversions.cc`**

```cpp
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>
#include <mptensor/index.hpp>

namespace {

using SIndex = mptensor::BasicIndex<std::ptrdiff_t>;
using mptensor::detail::UIndex;

TEST(NormalizeAxis, Boundaries) {
  EXPECT_EQ(mptensor::normalize_axis(-3, 3), 0u);
  EXPECT_EQ(mptensor::normalize_axis(-1, 3), 2u);
  EXPECT_EQ(mptensor::normalize_axis(0, 3), 0u);
  EXPECT_EQ(mptensor::normalize_axis(2, 3), 2u);
  EXPECT_THROW(mptensor::normalize_axis(-4, 3), std::out_of_range);
  EXPECT_THROW(mptensor::normalize_axis(3, 3), std::out_of_range);
}

TEST(NormalizeAxis, MessageNamesValueAndRange) {
  try {
    mptensor::normalize_axis(5, 3);
    FAIL() << "no exception";
  } catch (const std::out_of_range& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("axis 5"), std::string::npos) << msg;
    EXPECT_NE(msg.find("[-3, 3)"), std::string::npos) << msg;
  }
}

TEST(NormalizeAxes, ConvertsEachElementAndReportsPosition) {
  EXPECT_EQ(mptensor::normalize_axes(SIndex(-1, 0, -3), 4), UIndex(3, 0, 1));
  EXPECT_EQ(mptensor::normalize_axes(SIndex(), 4).size(), 0u);
  try {
    mptensor::normalize_axes(SIndex(0, 7), 4);
    FAIL() << "no exception";
  } catch (const std::out_of_range& e) {
    EXPECT_NE(std::string(e.what()).find("position 1"), std::string::npos)
        << e.what();
  }
}

TEST(NormalizeIndex, ScalarBoundaries) {
  EXPECT_EQ(mptensor::normalize_index(-5, 5), 0u);
  EXPECT_EQ(mptensor::normalize_index(-1, 5), 4u);
  EXPECT_EQ(mptensor::normalize_index(4, 5), 4u);
  EXPECT_THROW(mptensor::normalize_index(-6, 5), std::out_of_range);
  EXPECT_THROW(mptensor::normalize_index(5, 5), std::out_of_range);
}

TEST(NormalizeIndex, IndexVersion) {
  EXPECT_EQ(mptensor::normalize_index(SIndex(-1, 0), UIndex(3, 4)),
            UIndex(2, 0));
  EXPECT_THROW(mptensor::normalize_index(SIndex(0, 4), UIndex(3, 4)),
               std::out_of_range);
  EXPECT_THROW(mptensor::normalize_index(SIndex(0), UIndex(3, 4)),
               std::invalid_argument);
}

TEST(NormalizeSliceEnd, AcceptsSize) {
  EXPECT_EQ(mptensor::normalize_slice_end(5, 5), 5u);
  EXPECT_EQ(mptensor::normalize_slice_end(-5, 5), 0u);
  EXPECT_EQ(mptensor::normalize_slice_end(-1, 5), 4u);
  EXPECT_THROW(mptensor::normalize_slice_end(6, 5), std::out_of_range);
  EXPECT_THROW(mptensor::normalize_slice_end(-6, 5), std::out_of_range);
}

TEST(ToInternalShape, RejectsNegative) {
  EXPECT_EQ(mptensor::to_internal_shape(SIndex(2, 0, 3)), UIndex(2, 0, 3));
  EXPECT_THROW(mptensor::to_internal_shape(SIndex(2, -1)),
               std::invalid_argument);
}

TEST(ToPublic, RoundTrip) {
  const UIndex u(0, 7, 42);
  EXPECT_EQ(mptensor::to_internal_shape(mptensor::to_public(u)), u);
  UIndex huge;
  huge.push(std::numeric_limits<std::size_t>::max());
  EXPECT_THROW(mptensor::to_public(huge), std::out_of_range);
}

TEST(IdentityAxes, Sequence) {
  EXPECT_EQ(mptensor::detail::identity_axes(3), UIndex(0, 1, 2));
  EXPECT_EQ(mptensor::detail::identity_axes(0).size(), 0u);
}

TEST(NormalizeSliceRange, ScalarSlice) {
  using mptensor::detail::normalize_slice_range;
  EXPECT_EQ(normalize_slice_range(1, -1, 5, 0), std::make_pair<std::size_t, std::size_t>(1, 4));
  EXPECT_EQ(normalize_slice_range(-2, 5, 5, 0), std::make_pair<std::size_t, std::size_t>(3, 5));
  EXPECT_THROW(normalize_slice_range(2, -3, 5, 0), std::out_of_range);  // empty
  EXPECT_THROW(normalize_slice_range(3, 3, 5, 0), std::out_of_range);   // empty
  EXPECT_THROW(normalize_slice_range(5, 5, 5, 0), std::out_of_range);   // begin == n
}

TEST(NormalizeSliceRanges, IndexSlice) {
  UIndex b, e;
  mptensor::detail::normalize_slice_ranges(SIndex(0, 1, -2, 3), SIndex(0, -1, 5, 3),
                                           UIndex(5, 5, 5, 5), b, e);
  EXPECT_EQ(b, UIndex(0, 1, 3, 0));  // raw equal -> full axis
  EXPECT_EQ(e, UIndex(5, 4, 5, 5));
  EXPECT_THROW(mptensor::detail::normalize_slice_ranges(
                   SIndex(2), SIndex(-3), UIndex(5), b, e),
               std::out_of_range);
  EXPECT_THROW(mptensor::detail::normalize_slice_ranges(
                   SIndex(0, 0), SIndex(1), UIndex(5, 5), b, e),
               std::invalid_argument);
}

}  // namespace
```

- [ ] **Step 2: Register and run it to verify it fails**

In `tests/index/CMakeLists.txt`, change the first line to `add_executable(test_index basic_index.cc conversions.cc)`.
Run: `cmake --build build/si-mpi --target test_index 2>&1 | grep -m3 error`
Expected: errors such as `no member named 'normalize_axis' in namespace 'mptensor'`.

- [ ] **Step 3: Implement the conversion functions**

Append to `include/mptensor/index.hpp`, after the `range` overloads and before `//! \}`:

```cpp
namespace detail {
//! Normalize one value into [0, n) (or [0, n] if \c end_inclusive).
/*!
  \param position Element number shown in the error message; negative for a
  scalar argument.
*/
inline size_t normalize_value(std::ptrdiff_t v, size_t n, bool end_inclusive,
                              const char* kind, std::ptrdiff_t position) {
  const std::ptrdiff_t sn = static_cast<std::ptrdiff_t>(n);
  const bool ok = end_inclusive ? (v >= -sn && v <= sn) : (v >= -sn && v < sn);
  if (!ok) {
    std::ostringstream ss;
    ss << "mptensor: " << kind << " " << v;
    if (position >= 0) ss << " (position " << position << ")";
    ss << " is out of range [" << -sn << ", " << sn
       << (end_inclusive ? "]" : ")");
    throw std::out_of_range(ss.str());
  }
  return static_cast<size_t>(v < 0 ? v + sn : v);
}

//! [0, 1, ..., n-1]
inline UIndex identity_axes(size_t n) {
  UIndex axes;
  axes.resize(n);
  for (size_t i = 0; i < n; ++i) axes[i] = i;
  return axes;
}
}  // namespace detail

//! Normalize an axis: [-rank, rank) -> [0, rank).
inline size_t normalize_axis(std::ptrdiff_t a, size_t rank) {
  return detail::normalize_value(a, rank, false, "axis", -1);
}

//! Normalize each axis: [-rank, rank) -> [0, rank).
inline detail::UIndex normalize_axes(const BasicIndex<std::ptrdiff_t>& axes,
                                     size_t rank) {
  detail::UIndex result;
  result.resize(axes.size());
  for (size_t i = 0; i < axes.size(); ++i) {
    result[i] = detail::normalize_value(axes[i], rank, false, "axis",
                                        static_cast<std::ptrdiff_t>(i));
  }
  return result;
}

//! Normalize an element index: [-n, n) -> [0, n).
inline size_t normalize_index(std::ptrdiff_t i, size_t n) {
  return detail::normalize_value(i, n, false, "index", -1);
}

//! Normalize a global element index against \c shape.
inline detail::UIndex normalize_index(const BasicIndex<std::ptrdiff_t>& idx,
                                      const detail::UIndex& shape) {
  if (idx.size() != shape.size()) {
    std::ostringstream ss;
    ss << "mptensor: index " << idx << " has " << idx.size()
       << " elements, but the tensor has rank " << shape.size();
    throw std::invalid_argument(ss.str());
  }
  detail::UIndex result;
  result.resize(idx.size());
  for (size_t k = 0; k < idx.size(); ++k) {
    result[k] = detail::normalize_value(idx[k], shape[k], false, "index",
                                        static_cast<std::ptrdiff_t>(k));
  }
  return result;
}

//! Normalize an exclusive slice end: [-n, n] -> [0, n].
inline size_t normalize_slice_end(std::ptrdiff_t e, size_t n) {
  return detail::normalize_value(e, n, true, "slice end", -1);
}

//! Convert a public shape to the internal type. Negative sizes are invalid.
inline detail::UIndex to_internal_shape(const BasicIndex<std::ptrdiff_t>& s) {
  detail::UIndex result;
  result.resize(s.size());
  for (size_t k = 0; k < s.size(); ++k) {
    if (s[k] < 0) {
      std::ostringstream ss;
      ss << "mptensor: shape " << s << " has a negative size at position " << k;
      throw std::invalid_argument(ss.str());
    }
    result[k] = static_cast<size_t>(s[k]);
  }
  return result;
}

//! Convert an internal index to the public type.
inline BasicIndex<std::ptrdiff_t> to_public(const detail::UIndex& u) {
  BasicIndex<std::ptrdiff_t> result;
  result.resize(u.size());
  for (size_t k = 0; k < u.size(); ++k) {
    result[k] = detail::checked_index_cast<std::ptrdiff_t>(u[k]);
  }
  return result;
}

namespace detail {
//! Normalize a scalar slice [begin, end) on an axis of size \c n.
/*! \throw std::out_of_range if a bound is out of range or the slice is empty. */
inline std::pair<size_t, size_t> normalize_slice_range(std::ptrdiff_t begin,
                                                       std::ptrdiff_t end,
                                                       size_t n, size_t axis) {
  const std::ptrdiff_t pos = static_cast<std::ptrdiff_t>(axis);
  const size_t b = normalize_value(begin, n, false, "slice begin", pos);
  const size_t e = normalize_value(end, n, true, "slice end", pos);
  if (b >= e) {
    std::ostringstream ss;
    ss << "mptensor: slice [" << begin << ", " << end << ") on axis " << axis
       << " is empty (normalized to [" << b << ", " << e << "))";
    throw std::out_of_range(ss.str());
  }
  return {b, e};
}

//! Normalize Index-style slice bounds; raw begin[r] == end[r] means the full axis.
inline void normalize_slice_ranges(const BasicIndex<std::ptrdiff_t>& begin,
                                   const BasicIndex<std::ptrdiff_t>& end,
                                   const UIndex& shape, UIndex& ubegin,
                                   UIndex& uend) {
  const size_t rank = shape.size();
  if (begin.size() != rank || end.size() != rank) {
    std::ostringstream ss;
    ss << "mptensor: slice bounds " << begin << " and " << end
       << " do not match the tensor rank " << rank;
    throw std::invalid_argument(ss.str());
  }
  ubegin.resize(rank);
  uend.resize(rank);
  for (size_t r = 0; r < rank; ++r) {
    if (begin[r] == end[r]) {
      ubegin[r] = 0;
      uend[r] = shape[r];
    } else {
      const std::pair<size_t, size_t> be =
          normalize_slice_range(begin[r], end[r], shape[r], r);
      ubegin[r] = be.first;
      uend[r] = be.second;
    }
  }
}
}  // namespace detail
```

- [ ] **Step 4: Run the tests and the whole suite**

Run: `cmake --build build/si-mpi -j2 && cmake --build build/si-nompi -j2 && ctest --test-dir build/si-mpi --output-on-failure && ctest --test-dir build/si-nompi --output-on-failure`
Expected: all pass (17 and 5 tests); `mpiexec -n 1 build/si-mpi/tests/index/test_index --gtest_brief=1` reports all `NormalizeAxis*`, `NormalizeIndex*`, ... tests passed.

- [ ] **Step 5: Commit**

```bash
git add include/mptensor/index.hpp tests/index
git commit -m "Add conversion functions between public and internal indices

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Make the public types signed and normalize at every entry point

This is one cohesive change: once `Index` becomes signed, the whole library must be converted before anything compiles. Write the tensor-level tests first, then convert file by file, then build.

**Conversion rules** (apply them everywhere this task touches; the steps below list every function):

| In internal code | becomes |
|---|---|
| local `Index`/`Axes`/`Shape` holding internal values | `detail::UIndex` |
| `x.shape()` used for computation | `x.internal_shape()` |
| `transpose(t, axes, urank)` with internal `axes` | `detail::transpose_impl(t, axes, urank)` |
| `reshape(t, shape)` with internal `shape` | `detail::reshape_impl(t, shape)` |
| `Tensor<M>(comm, shape, urank)` / `Tensor<M> x(comm, shape, urank)` with internal `shape` | add `, detail::internal` as the last argument |
| `Tensor<M> x(comm, shape)` with internal `shape` | `Tensor<M> x(comm, shape, shape.size() / 2, detail::internal)` |
| literal `Shape(a, b)` / `Axes(a, b)` passed to the `_impl` functions or the tagged constructor | `detail::UIndex(a, b)` |
| `range(n)` compared with or assigned to `axes_map` | `detail::identity_axes(n)` |

Calls to *public* functions with literal non-negative arguments (e.g. `svd(a, Axes(0), Axes(1), s)`, `slice(u, 1, 0, target_rank)`, `tensordot(..., Axes(n), Axes(0))`, `range(0, rank_row)` in `rsvd`) stay as they are: they go through normalization once, which is a no-op for non-negative values.

**Files:**
- Create: `tests/tensor/negative_index.cc`; Modify: `tests/tensor/CMakeLists.txt`
- Modify: `include/mptensor/index.hpp`, `include/mptensor/tensor.hpp`, `include/mptensor/tensor_impl.hpp`, `include/mptensor/rsvd_impl.hpp`, `include/mptensor/file_io/load.hpp`, `src/tensor.cc`

**Interfaces:**
- Consumes: everything from Tasks 1–2.
- Produces:
  - `using Index = BasicIndex<std::ptrdiff_t>;` ; `Index range(std::ptrdiff_t start, std::ptrdiff_t stop)` (throws `std::invalid_argument` if `start > stop`), `Index range(std::ptrdiff_t stop)`.
  - `namespace detail { struct internal_t { explicit internal_t() = default; }; inline constexpr internal_t internal{}; }` (in `tensor.hpp`).
  - `Tensor(const comm_type&, const detail::UIndex& shape, size_t upper_rank, detail::internal_t)`.
  - `Shape Tensor::shape() const`; `const detail::UIndex& Tensor::internal_shape() const`; `const detail::UIndex& Tensor::get_axes_map() const`.
  - `void global_index_fast(size_t, detail::UIndex&) const`; `void local_position(const detail::UIndex&, int&, size_t&) const`.
  - `template <typename M> Tensor<M> detail::transpose_impl(const Tensor<M>&, const detail::UIndex& axes, size_t urank)`; `template <typename M> Tensor<M> detail::reshape_impl(const Tensor<M>&, const detail::UIndex& shape)`.
  - Scalar arguments `n_axes`, `i_begin`, `i_end` are `std::ptrdiff_t`.

- [ ] **Step 1: Write the failing test `tests/tensor/negative_index.cc`**

```cpp
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

// A[L, L+1, L+2, L+3] filled with func4_1.
template <typename TensorType>
TensorType make_rank4() {
  using T = typename TensorType::value_type;
  TensorType A(Shape(kL, kL + 1, kL + 2, kL + 3));
  fill_tensor(A, func4_1<T>);
  return A;
}

template <typename TensorType>
double max_diff(const TensorType& x, const TensorType& y) {
  return max_abs(x - y);
}

template <typename TensorType>
class NegativeIndex : public ::testing::Test {};
TYPED_TEST_SUITE(NegativeIndex, TensorTypes, TensorTypeNames);

TYPED_TEST(NegativeIndex, TransposeFreeAndMember) {
  const TypeParam A = make_rank4<TypeParam>();
  const TypeParam expected = transpose(A, Axes(2, 0, 3, 1));
  EXPECT_EQ(max_diff(transpose(A, Axes(-2, 0, -1, 1)), expected), 0.0);
  EXPECT_EQ(max_diff(transpose(A, Axes(-2, -4, -1, -3), 2),
                     transpose(A, Axes(2, 0, 3, 1), 2)),
            0.0);
  TypeParam B = A;
  B.transpose(Axes(2, -4, 3, -3));
  EXPECT_EQ(max_diff(B, expected), 0.0);
  EXPECT_EQ(max_diff(transpose(A, range(-4, 0)), A), 0.0);
}

TYPED_TEST(NegativeIndex, Tensordot) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 4, kL + 1, kL + 5));
  TypeParam B(Shape(kL + 5, kL + 2, kL + 4, kL + 3));
  fill_tensor(A, func4_1<T>);
  fill_tensor(B, func4_2<T>);
  EXPECT_EQ(max_diff(tensordot(A, B, Axes(-3, -1), Axes(2, -4)),
                     tensordot(A, B, Axes(1, 3), Axes(2, 0))),
            0.0);
}

TYPED_TEST(NegativeIndex, ContractAndTrace) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 1, kL, kL + 2));
  fill_tensor(A, func4_1<T>);
  EXPECT_EQ(max_diff(contract(A, -4, -2), contract(A, 0, 2)), 0.0);

  TypeParam C(Shape(kL, kL + 1, kL + 1, kL));
  fill_tensor(C, func4_1<T>);
  EXPECT_EQ(std::abs(trace(C, Axes(0, -3), Axes(-1, 2)) -
                     trace(C, Axes(0, 1), Axes(3, 2))),
            0.0);

  TypeParam D(Shape(kL + 2, kL + 1, kL + 3, kL));
  fill_tensor(D, func4_2<T>);
  TypeParam E(Shape(kL, kL + 1, kL + 2, kL + 3));
  fill_tensor(E, func4_1<T>);
  EXPECT_EQ(std::abs(trace(E, D, Axes(0, -1, 2, -3), Axes(-1, 2, 0, -3)) -
                     trace(E, D, Axes(0, 3, 2, 1), Axes(3, 2, 0, 1))),
            0.0);
}

TYPED_TEST(NegativeIndex, SvdQrEighMultiplyVector) {
  using T = typename TypeParam::value_type;
  const TypeParam A = make_rank4<TypeParam>();

  TypeParam U1, V1, U2, V2;
  std::vector<double> S1, S2;
  svd(A, Axes(-2, 0), Axes(1, -1), U1, S1, V1);
  svd(A, Axes(2, 0), Axes(1, 3), U2, S2, V2);
  EXPECT_EQ(S1, S2);
  EXPECT_EQ(max_diff(U1, U2), 0.0);
  EXPECT_EQ(max_diff(V1, V2), 0.0);

  TypeParam U3 = U2;
  U2.multiply_vector(S2, 2);
  U3.multiply_vector(S2, -1);
  EXPECT_EQ(max_diff(U2, U3), 0.0);

  TypeParam Q1, R1, Q2, R2;
  qr(A, Axes(-2, -4), Axes(-3, -1), Q1, R1);
  qr(A, Axes(2, 0), Axes(1, 3), Q2, R2);
  EXPECT_EQ(max_diff(Q1, Q2), 0.0);
  EXPECT_EQ(max_diff(R1, R2), 0.0);

  TypeParam H(Shape(kL, kL, kL, kL));
  fill_tensor(H, [](const Index& idx, const Shape& shape) {
    return func4_1<T>(idx, shape) +
           conj_value(func4_1<T>(Index(idx[2], idx[3], idx[0], idx[1]), shape));
  });
  TypeParam Z1, Z2;
  std::vector<double> W1, W2;
  eigh(H, Axes(-4, -3), Axes(2, -1), W1, Z1);
  eigh(H, Axes(0, 1), Axes(2, 3), W2, Z2);
  EXPECT_EQ(W1, W2);
  EXPECT_EQ(max_diff(Z1, Z2), 0.0);
}

TYPED_TEST(NegativeIndex, SliceAndSetSlice) {
  using T = typename TypeParam::value_type;
  const TypeParam A = make_rank4<TypeParam>();  // (10, 11, 12, 13)
  EXPECT_EQ(max_diff(slice(A, -3, 3, -5), slice(A, 1, 3, 6)), 0.0);
  EXPECT_EQ(max_diff(slice(A, Index(0, 2, 0, -12), Index(0, -7, 0, 5)),
                     slice(A, Index(0, 2, 0, 1), Index(0, 4, 0, 5))),
            0.0);
  // Raw begin == end keeps the whole axis, also for a nonzero value.
  EXPECT_EQ(max_diff(slice(A, Index(3, 2, -1, 0), Index(3, 4, -1, 0)),
                     slice(A, Index(0, 2, 0, 0), Index(0, 4, 0, 0))),
            0.0);

  TypeParam X = A;
  TypeParam Y = A;
  TypeParam B(Shape(kL, 3, kL + 2, kL + 3));
  TypeParam C(Shape(3, 2, kL + 2, 3));
  fill_tensor(B, func4_2<T>);
  fill_tensor(C, func4_2<T>);
  X.set_slice(B, -3, 3, -5);
  Y.set_slice(B, 1, 3, 6);
  X.set_slice(C, Index(-9, 0, 0, -11), Index(4, -9, 0, -8));
  Y.set_slice(C, Index(1, 0, 0, 2), Index(4, 2, 0, 5));
  EXPECT_EQ(max_diff(X, Y), 0.0);
}

TYPED_TEST(NegativeIndex, GetAndSetValue) {
  using T = typename TypeParam::value_type;
  TypeParam X(Shape(kL, kL + 1, kL + 2, kL + 3));
  TypeParam Y(Shape(kL, kL + 1, kL + 2, kL + 3));
  X = 0.0;
  Y = 0.0;
  X.set_value(Index(-1, -2, 0, -1), T(5.0));
  Y.set_value(Index(kL - 1, kL - 1, 0, kL + 2), T(5.0));
  EXPECT_EQ(max_diff(X, Y), 0.0);

  T value = 0.0;
  const bool mine = X.get_value(Index(-1, -2, 0, -1), value);
  if (mine) EXPECT_EQ(value, T(5.0));
  EXPECT_EQ(max_over_ranks(mine ? 1.0 : 0.0, X.get_comm()), 1.0);
}

template <typename TensorType>
class NegativeIndexExceptions : public ::testing::Test {};
TYPED_TEST_SUITE(NegativeIndexExceptions, TensorTypes, TensorTypeNames);

TYPED_TEST(NegativeIndexExceptions, OutOfRangeAxes) {
  const TypeParam A = make_rank4<TypeParam>();
  EXPECT_THROW(transpose(A, Axes(0, 1, 2, 4)), std::out_of_range);
  EXPECT_THROW(transpose(A, Axes(-5, 1, 2, 3), 2), std::out_of_range);
  EXPECT_THROW(tensordot(A, A, Axes(1, 4), Axes(1, 3)), std::out_of_range);
  TypeParam U, V;
  std::vector<double> S;
  EXPECT_THROW(svd(A, Axes(0, 1), Axes(2, -5), U, S, V), std::out_of_range);
  EXPECT_THROW(slice(A, 4, 0, 1), std::out_of_range);
}

TYPED_TEST(NegativeIndexExceptions, OutOfRangeIndices) {
  using T = typename TypeParam::value_type;
  TypeParam A = make_rank4<TypeParam>();  // (10, 11, 12, 13)
  T value;
  EXPECT_THROW(A.get_value(Index(0, 0, 0, 13), value), std::out_of_range);
  EXPECT_THROW(A.set_value(Index(-11, 0, 0, 0), T(1.0)), std::out_of_range);
  EXPECT_THROW(A.get_value(Index(0, 0, 0), value), std::invalid_argument);
  EXPECT_THROW(slice(A, 1, 0, 12), std::out_of_range);  // end > n
  EXPECT_THROW(slice(A, 1, 3, 3), std::out_of_range);   // empty
  EXPECT_THROW(slice(A, Index(0, 2, 0, 0), Index(0, -9, 0, 0)),
               std::out_of_range);  // [2, 2) is empty
}

TYPED_TEST(NegativeIndexExceptions, NegativeShape) {
  const TypeParam A = make_rank4<TypeParam>();
  EXPECT_THROW(TypeParam(Shape(2, -1)), std::invalid_argument);
  EXPECT_THROW(reshape(A, Shape(-1, (kL + 2) * (kL + 3))),
               std::invalid_argument);
}

TYPED_TEST(NegativeIndexExceptions, TensorUnchangedAfterThrow) {
  using T = typename TypeParam::value_type;
  TypeParam A = make_rank4<TypeParam>();
  const TypeParam A0 = A;
  const Shape shape0 = A.shape();

  EXPECT_THROW(A.transpose(Axes(0, 1, 2, 9)), std::out_of_range);
  TypeParam B(Shape(kL, 3, kL + 2, kL + 3));
  EXPECT_THROW(A.set_slice(B, 1, 3, 99), std::out_of_range);
  std::vector<double> v(kL + 3, 2.0);
  EXPECT_THROW(A.multiply_vector(v, 4), std::out_of_range);
  EXPECT_THROW(A.set_value(Index(0, 0, 0, 99), T(1.0)), std::out_of_range);

  EXPECT_EQ(A.shape(), shape0);
  EXPECT_EQ(max_diff(A, A0), 0.0);
}

TYPED_TEST(NegativeIndexExceptions, CollectiveContinuesAfterThrow) {
  const TypeParam A = make_rank4<TypeParam>();
  EXPECT_THROW(tensordot(A, A, Axes(0, 9), Axes(0, 1)), std::out_of_range);
  // Every rank threw at the same point, so the next collective call matches.
  const TypeParam C = tensordot(A, A, Axes(0, 1), Axes(0, 1));
  EXPECT_EQ(C.shape(), Shape(kL + 2, kL + 3, kL + 2, kL + 3));
  EXPECT_GT(max_abs(C), 0.0);
}

}  // namespace
}  // namespace mptensor_test
```

- [ ] **Step 2: Register it and run it to verify it fails**

In `tests/tensor/CMakeLists.txt`, after the last `target_sources` line, add `target_sources(test_tensor PRIVATE negative_index.cc)`.
Run: `cmake --build build/si-mpi --target test_tensor 2>&1 | grep -m3 error; mpiexec -n 1 build/si-mpi/tests/tensor/test_tensor --gtest_filter='NegativeIndex*' 2>&1 | tail -5`
Expected: either compile errors (e.g. no `slice(..., int, int, int)` accepting negative) or a run that FAILS/aborts — `Axes(0, -1)` currently throws `std::out_of_range` while constructing a `size_t` index, `Shape(2, -1)` throws `out_of_range` instead of `invalid_argument`, and out-of-range axes hit `assert`. Do not continue until you have seen it fail.

- [ ] **Step 3: Flip the public type in `include/mptensor/index.hpp`**

Replace

```cpp
using Index = BasicIndex<std::size_t>;
```

with

```cpp
using Index = BasicIndex<std::ptrdiff_t>;  //!< Public index type (also Axes and Shape).
```

and replace the two `range` overloads with:

```cpp
//! Create an increasing sequence [start, stop). It is similar to range() in python.
/*! \throw std::invalid_argument if <tt>start > stop</tt>. */
inline Index range(const std::ptrdiff_t start, const std::ptrdiff_t stop) {
  if (start > stop) {
    std::ostringstream ss;
    ss << "mptensor: range(" << start << ", " << stop << ") has start > stop";
    throw std::invalid_argument(ss.str());
  }
  Index index;
  index.resize(static_cast<size_t>(stop - start));
  for (std::ptrdiff_t i = start; i < stop; ++i) {
    index[static_cast<size_t>(i - start)] = i;
  }
  return index;
}
inline Index range(const std::ptrdiff_t stop) { return range(0, stop); }
```

Move these `range` overloads *after* the conversion functions block (they now need `<sstream>` only, which is already included). In `tests/index/basic_index.cc`, the `Range` test keeps passing; add to it:

```cpp
  EXPECT_EQ(mptensor::range(-2, 1), mptensor::Index(-2, -1, 0));
  EXPECT_THROW(mptensor::range(3, 2), std::invalid_argument);
```

- [ ] **Step 4: Update declarations in `include/mptensor/tensor.hpp`**

1. After `using Shape = Index;` add:

```cpp
namespace detail {
//! Tag for the internal constructor that takes a detail::UIndex shape.
struct internal_t {
  explicit internal_t() = default;
};
inline constexpr internal_t internal{};
}  // namespace detail
```

2. In the class, constructors: after `Tensor(const comm_type &, const Shape &, size_t upper_rank);` add

```cpp
  //! \cond
  Tensor(const comm_type &, const detail::UIndex &shape, size_t upper_rank,
         detail::internal_t);  // internal use: shape is already normalized
  //! \endcond
```

3. Replace these member declarations:

| Before | After |
|---|---|
| `const Shape &shape() const;` | `Shape shape() const;` and on the next line `const detail::UIndex &internal_shape() const;  //!< Shape as the internal type.` |
| `const Axes &get_axes_map() const;` | `const detail::UIndex &get_axes_map() const;` |
| `void global_index_fast(size_t i, Index &idx) const;` | `void global_index_fast(size_t i, detail::UIndex &idx) const;` |
| `void local_position(const Index &idx, int &comm_rank, size_t &local_idx) const;` | `void local_position(const detail::UIndex &idx, int &comm_rank, size_t &local_idx) const;` |
| every `size_t n_axes`, `size_t n_axes0` … `size_t n_axes3` in `multiply_vector` | `std::ptrdiff_t n_axes` … |
| `set_slice(const Tensor &a, size_t n_axes, size_t i_begin, size_t i_end);` | `set_slice(const Tensor &a, std::ptrdiff_t n_axes, std::ptrdiff_t i_begin, std::ptrdiff_t i_end);` |
| `Shape Dim;` | `detail::UIndex Dim;` |
| `Axes axes_map;` | `detail::UIndex axes_map;` |
| `void init(const Shape &, size_t upper_rank);` | `void init(const detail::UIndex &, size_t upper_rank);` |
| `void init(const Shape &, size_t upper_rank, const Axes &map);` | `void init(const detail::UIndex &, size_t upper_rank, const detail::UIndex &map);` |
| `void change_configuration(const size_t new_upper_rank, const Axes &new_axes_map);` | `void change_configuration(const size_t new_upper_rank, const detail::UIndex &new_axes_map);` |
| `bool local_index(const Index &, size_t &i) const;` | `bool local_index(const detail::UIndex &, size_t &i) const;` |

Also in the private section add `Tensor<MatrixType> &transpose_internal(const detail::UIndex &axes);  // axes already normalized`.

4. Free-function declarations: `slice(const Tensor<MatrixType> &a, size_t n_axes, size_t i_begin, size_t i_end)` → `std::ptrdiff_t` for all three. After the `extend` declaration add:

```cpp
namespace detail {
template <typename MatrixType>
Tensor<MatrixType> transpose_impl(const Tensor<MatrixType> &a, const UIndex &axes,
                                  size_t urank_new);
template <typename MatrixType>
Tensor<MatrixType> reshape_impl(const Tensor<MatrixType> &a, const UIndex &shape_new);
}  // namespace detail
```

5. Add a Doxygen note to the class documentation (`/*! Tensor class ... */`):

```cpp
  Axes and element indices accept negative values, which count from the end
  as in numpy. Out-of-range values throw std::out_of_range, and negative sizes
  in a Shape throw std::invalid_argument, before any communication starts.
  With MPI, collective operations must receive the same arguments on all
  processes; then all processes throw together and may catch and continue.
```

- [ ] **Step 5: Convert `src/tensor.cc` and the declarations at the top of `tensor_impl.hpp`**

In `src/tensor.cc` and in the forward declarations at `tensor_impl.hpp:50-60`, change every parameter type `const Axes&` and `const Shape&` of `is_no_transpose` and of all `debug::check_*` functions to `const detail::UIndex&`, and every local `Axes v` / `Axes axes` inside them to `detail::UIndex`. No logic changes. Example:

```cpp
bool check_transpose_axes(const detail::UIndex& axes, size_t rank) {
  if (axes.size() != rank) return false;
  detail::UIndex v = axes;
  v.sort();
  for (size_t i = 0; i < rank; ++i) {
    if (v[i] != i) return false;
  }
  return true;
}
```

- [ ] **Step 6: Convert the `Tensor` members in `tensor_impl.hpp`**

Replace the definitions below (doc comments stay; add `\throw` lines where noted).

Constructors:

```cpp
template <typename MatrixType>
Tensor<MatrixType>::Tensor(const Shape &shape) : Mat() {
  init(to_internal_shape(shape), shape.size() / 2);
};

template <typename MatrixType>
Tensor<MatrixType>::Tensor(const comm_type &comm, const Shape &shape)
    : Mat(comm) {
  init(to_internal_shape(shape), shape.size() / 2);
};

template <typename MatrixType>
Tensor<MatrixType>::Tensor(const comm_type &comm, const Shape &shape,
                          const size_t upper_rank)
    : Mat(comm) {
  init(to_internal_shape(shape), upper_rank);
};

//! \cond
template <typename MatrixType>
Tensor<MatrixType>::Tensor(const comm_type &comm, const detail::UIndex &shape,
                          const size_t upper_rank, detail::internal_t)
    : Mat(comm) {
  init(shape, upper_rank);
};
//! \endcond

template <typename MatrixType>
Tensor<MatrixType>::Tensor(const comm_type &comm,
                          const Tensor<lapack::Matrix<value_type>> &t)
    : Mat(comm) {
  init(t.internal_shape(), t.get_upper_rank());
  const size_t n = Mat.local_size();
  size_t idx;
  int dummy;
  detail::UIndex g;
  g.resize(Dim.size());
  for (size_t i = 0; i < n; ++i) {
    global_index_fast(i, g);
    t.local_position(g, dummy, idx);
    Mat[i] = t[idx];
  }
};

template <typename MatrixType>
Tensor<MatrixType>::Tensor(const comm_type &comm, const std::vector<value_type> &v)
    : Mat(comm) {
  init(detail::UIndex(v.size()), 0);
  const size_t n = Mat.local_size();
  detail::UIndex idx;
  idx.resize(1);
  for (size_t i = 0; i < n; ++i) {
    global_index_fast(i, idx);
    Mat[i] = v[idx[0]];
  }
};
```

Accessors:

```cpp
template <typename MatrixType>
inline Shape Tensor<MatrixType>::shape() const {
  return to_public(Dim);
}

template <typename MatrixType>
inline const detail::UIndex &Tensor<MatrixType>::internal_shape() const {
  return Dim;
}

template <typename MatrixType>
inline const detail::UIndex &Tensor<MatrixType>::get_axes_map() const {
  return axes_map;
}
```

`init` (both overloads):

```cpp
template <typename MatrixType>
inline void Tensor<MatrixType>::init(const detail::UIndex &shape, size_t urank) {
  init(shape, urank, detail::identity_axes(shape.size()));
}

template <typename MatrixType>
void Tensor<MatrixType>::init(const detail::UIndex &shape, size_t urank,
                             const detail::UIndex &map) {
  // body unchanged
}
```

`local_index`, `global_index_fast`, `local_position`: change only the parameter types (`const detail::UIndex &gindex`, `detail::UIndex &gindex`, `const detail::UIndex &index`); bodies unchanged.

`global_index`:

```cpp
template <typename MatrixType>
Index Tensor<MatrixType>::global_index(size_t lindex) const {
  detail::UIndex gindex;
  gindex.resize(Dim.size());
  global_index_fast(lindex, gindex);
  return to_public(gindex);
};
```

`get_value` / `set_value` (add `\throw std::out_of_range ...` and `\throw std::invalid_argument ...` to their docs):

```cpp
template <typename MatrixType>
bool Tensor<MatrixType>::get_value(const Index &idx, value_type &val) const {
  size_t li;
  if (local_index(normalize_index(idx, Dim), li)) {
    val = Mat[li];
    return true;
  } else {
    return false;
  }
}

template <typename MatrixType>
void Tensor<MatrixType>::set_value(const Index &idx, value_type val) {
  size_t li;
  if (local_index(normalize_index(idx, Dim), li)) {
    Mat[li] = val;
  }
}
```

`change_configuration`: signature `(const size_t new_upper_rank, const detail::UIndex &new_axes_map)`; locals `Shape dim; Axes axes;` → `detail::UIndex dim; detail::UIndex axes;`; `transpose(axes);` → `transpose_internal(axes);`; `Index index;` in the OpenMP block → `detail::UIndex index;`.

Member `transpose` and the new private `transpose_internal`:

```cpp
template <typename MatrixType>
Tensor<MatrixType> &Tensor<MatrixType>::transpose(const Axes &axes) {
  return transpose_internal(normalize_axes(axes, Dim.size()));
}

//! \cond
template <typename MatrixType>
Tensor<MatrixType> &Tensor<MatrixType>::transpose_internal(
    const detail::UIndex &axes) {
  const size_t rank = Dim.size();
  assert(debug::check_transpose_axes(axes, rank));

  detail::UIndex dim_now = Dim;
  detail::UIndex map_now = axes_map;
  detail::UIndex axes_inv;
  axes_inv.resize(rank);
  for (size_t i = 0; i < rank; ++i) {
    axes_inv[axes[i]] = i;
  }
  for (size_t i = 0; i < rank; ++i) {
    Dim[i] = dim_now[axes[i]];
    axes_map[i] = axes_inv[map_now[i]];
  }
  return (*this);
}
//! \endcond
```

`multiply_vector` (all four overloads): change each `size_t n_axesK` parameter to `std::ptrdiff_t n_axesK`; as the **first** statement of the body add `const size_t axK = normalize_axis(n_axesK, rank());` for every K; then replace every use of `n_axesK` in the body by `axK` and `Index idx;` by `detail::UIndex idx;`. For the single-vector version this gives:

```cpp
template <typename MatrixType>
template <typename D>
Tensor<MatrixType> &Tensor<MatrixType>::multiply_vector(const std::vector<D> &vec,
                                                      std::ptrdiff_t n_axes) {
  const size_t ax = normalize_axis(n_axes, rank());
  assert(Dim[ax] <= vec.size());
  const size_t local_size = this->local_size();
  prep_local_to_global();
#pragma omp parallel default(shared)
  {
    detail::UIndex idx;
    idx.resize(rank());
#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      global_index_fast(i, idx);
      Mat[i] *= vec[idx[ax]];
    }
  }
  return (*this);
}
```

(Keep the existing body structure of each overload; only the three substitutions above.)

`set_slice` scalar version:

```cpp
template <typename MatrixType>
Tensor<MatrixType> &Tensor<MatrixType>::set_slice(const Tensor<MatrixType> &a,
                                                const std::ptrdiff_t n_axes,
                                                const std::ptrdiff_t i_begin,
                                                const std::ptrdiff_t i_end) {
  const size_t ax = normalize_axis(n_axes, rank());
  const std::pair<size_t, size_t> be =
      detail::normalize_slice_range(i_begin, i_end, Dim[ax], ax);
  const size_t begin = be.first;
  const size_t end = be.second;
  assert(rank() == a.rank());
  assert(end - begin == a.internal_shape()[ax]);

  /* create lists of local position and destination rank */
  const size_t local_size = a.local_size();
  std::vector<int> dest_mpirank(local_size);
  std::vector<size_t> local_position(local_size);

  a.prep_local_to_global();
  prep_global_to_local();

#pragma omp parallel default(shared)
  {
    detail::UIndex index;
    index.resize(rank());
#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      a.global_index_fast(i, index);
      index[ax] += begin;
      int dest;
      size_t pos;
      this->local_position(index, dest, pos);
      local_position[i] = pos;
      dest_mpirank[i] = dest;
    }
  }

  /* exchange data */
  replace_matrix_data(a.get_matrix(), dest_mpirank, local_position, Mat);
  return (*this);
}
```

Before editing, compare with the current body (`tensor_impl.hpp`, `set_slice` scalar version) and keep any statement not shown here (e.g. the exact `replace_matrix_data` call) unchanged.

`set_slice` `Index` version: replace the asserts at the top with

```cpp
  const size_t nr = rank();
  assert(nr == a.rank());
  detail::UIndex begin, end;
  detail::normalize_slice_ranges(index_begin, index_end, Dim, begin, end);
```

replace `Index index;` by `detail::UIndex index;`, and replace the per-axis offset line `if (index_begin[r] != index_end[r]) index[r] += index_begin[r];` by `index[r] += begin[r];` (a full axis has `begin[r] == 0`). `end` is only used by the assert `assert(end[r] - begin[r] == a.internal_shape()[r]);`, which you add inside a loop over `r` right after the normalization.

`gather` and `flatten`: replace `range(n)` (two places each) by `detail::identity_axes(n)`, and in `gather` replace `return reshape(T, Dim);` by `return detail::reshape_impl(T, Dim);`.

- [ ] **Step 7: Convert the shape-changing free functions**

`transpose` (3 arguments) and the new `detail::transpose_impl`:

```cpp
template <typename MatrixType>
Tensor<MatrixType> transpose(const Tensor<MatrixType> &T, const Axes &axes,
                            size_t urank_new) {
  return detail::transpose_impl(T, normalize_axes(axes, T.rank()), urank_new);
}

namespace detail {
template <typename MatrixType>
Tensor<MatrixType> transpose_impl(const Tensor<MatrixType> &T, const UIndex &axes,
                                  size_t urank_new) {
  // former body of transpose(T, axes, urank_new), with these substitutions:
  //   Shape dim_old = T.shape();   -> UIndex dim_old = T.internal_shape();
  //   Shape dim_new;               -> UIndex dim_new;
  //   Tensor<MatrixType> T_new(T.get_comm(), dim_new, urank_new);
  //                                -> ... (T.get_comm(), dim_new, urank_new, internal);
  //   Axes axes_map = T.get_axes_map(); -> UIndex axes_map = T.get_axes_map();
}
}  // namespace detail
```

The 2-argument `transpose(Tensor T, const Axes &axes)` stays `return T.transpose(axes);`.

`reshape` and `detail::reshape_impl`:

```cpp
template <typename MatrixType>
Tensor<MatrixType> reshape(const Tensor<MatrixType> &T, const Shape &shape_new) {
  return detail::reshape_impl(T, to_internal_shape(shape_new));
}

namespace detail {
template <typename MatrixType>
Tensor<MatrixType> reshape_impl(const Tensor<MatrixType> &T, const UIndex &shape_new) {
  // former body of reshape, with:
  //   assert(debug::check_total_size(shape_new, T.internal_shape()));
  //   const UIndex &shape = T.internal_shape();
  //   Tensor<MatrixType> T_new(T.get_comm(), shape_new, shape_new.size() / 2, internal);
  //   UIndex index, index_new;
}
}  // namespace detail
```

`slice` scalar version — replace the head of the function down to `Tensor<MatrixType> T_new(...)` by:

```cpp
template <typename MatrixType>
Tensor<MatrixType> slice(const Tensor<MatrixType> &T, std::ptrdiff_t n_axes,
                        std::ptrdiff_t i_begin, std::ptrdiff_t i_end) {
  const int mpisize = T.get_comm_size();
  const detail::UIndex &shape = T.internal_shape();
  const size_t ax = normalize_axis(n_axes, T.rank());
  const std::pair<size_t, size_t> be =
      detail::normalize_slice_range(i_begin, i_end, shape[ax], ax);
  const size_t begin = be.first;
  const size_t end = be.second;

  detail::UIndex shape_new = shape;
  shape_new[ax] = end - begin;

  /* initialize new tensor */
  Tensor<MatrixType> T_new(T.get_comm(), shape_new, shape_new.size() / 2,
                           detail::internal);
```

then in the rest of the body: `Index index;` → `detail::UIndex index;`, `n_axes` → `ax`, `i_begin` → `begin`, `i_end` → `end`.

`slice` `Index` version — replace the head down to `Tensor<MatrixType> T_new(...)` by:

```cpp
template <typename MatrixType>
Tensor<MatrixType> slice(const Tensor<MatrixType> &T, const Index &index_begin,
                        const Index &index_end) {
  const int mpisize = T.get_comm_size();
  const detail::UIndex &shape = T.internal_shape();
  const size_t rank = T.rank();
  detail::UIndex begin, end;
  detail::normalize_slice_ranges(index_begin, index_end, shape, begin, end);

  detail::UIndex shape_new;
  shape_new.resize(rank);
  for (size_t r = 0; r < rank; ++r) shape_new[r] = end[r] - begin[r];

  /* initialize new tensor */
  Tensor<MatrixType> T_new(T.get_comm(), shape_new, shape_new.size() / 2,
                           detail::internal);
```

and replace the per-axis test inside the OpenMP loop by:

```cpp
      for (size_t r = 0; r < rank; ++r) {
        const size_t idx = index[r];
        if (idx >= begin[r] && idx < end[r]) {
          index[r] -= begin[r];
        } else {
          is_send = false;
          break;
        }
      }
```

with `Index index;` → `detail::UIndex index;`.

`extend`:

```cpp
template <typename MatrixType>
Tensor<MatrixType> extend(const Tensor<MatrixType> &T, const Shape &shape_new_public) {
  const detail::UIndex shape_new = to_internal_shape(shape_new_public);
  assert(T.rank() == shape_new.size());
  assert(debug::check_extend(T.internal_shape(), shape_new));

  Tensor<MatrixType> T_new(T.get_comm(), shape_new, shape_new.size() / 2,
                           detail::internal);
  // rest unchanged, except `Index index;` -> `detail::UIndex index;`
```

- [ ] **Step 8: Convert trace, contract, kron, tensordot**

`trace(T, axes_1, axes_2)`: as the first statements

```cpp
  const detail::UIndex ax1 = normalize_axes(axes_1, T.rank());
  const detail::UIndex ax2 = normalize_axes(axes_2, T.rank());
```

then use `ax1`/`ax2` everywhere instead of `axes_1`/`axes_2` (including the asserts), and `Index index;` → `detail::UIndex index;`.

`trace(A, B, axes_a, axes_b)`: first statements

```cpp
  const detail::UIndex ax_a = normalize_axes(axes_a, A.rank());
  const detail::UIndex ax_b = normalize_axes(axes_b, B.rank());
```

use them instead of `axes_a`/`axes_b`; the assert becomes `debug::check_trace_axes(ax_a, ax_b, A.internal_shape(), B.internal_shape())`; `Axes axes; Axes axes_a_inv; Axes axes_map = ...` → `detail::UIndex`; `transpose(B, axes, A.get_upper_rank())` → `detail::transpose_impl(B, axes, A.get_upper_rank())`.

`contract(T, axes_1, axes_2)`: first statements after `mpisize`

```cpp
  const detail::UIndex ax1 = normalize_axes(axes_1, T.rank());
  const detail::UIndex ax2 = normalize_axes(axes_2, T.rank());
```

use them instead of `axes_1`/`axes_2`; `Shape shape = T.shape(); Shape shape_new; Axes axes_new;` → `detail::UIndex shape = T.internal_shape(); detail::UIndex shape_new; detail::UIndex axes_new;`; `Axes v = ...` → `detail::UIndex v = ax1 + ax2;`; `Tensor<MatrixType> T_new(T.get_comm(), shape_new);` → tagged constructor with `shape_new.size() / 2`; `Index index, index_new;` → `detail::UIndex index, index_new;`.

`kron(a, b)`:

```cpp
template <typename MatrixType>
Tensor<MatrixType> kron(const Tensor<MatrixType> &a, const Tensor<MatrixType> &b) {
  assert(a.rank() == b.rank());
  assert(a.get_comm() == b.get_comm());

  const detail::UIndex shape_a = a.internal_shape();
  const detail::UIndex shape_b = b.internal_shape();
  detail::UIndex shape_c = shape_a;
  const size_t n = shape_a.size();
  detail::UIndex axes_trans;
  axes_trans.resize(2 * n);
  for (size_t i = 0; i < shape_b.size(); ++i) {
    shape_c[i] *= shape_b[i];
    axes_trans[2 * i] = i;
    axes_trans[2 * i + 1] = i + n;
  }

  Tensor<MatrixType> ab =
      tensordot(detail::reshape_impl(a, shape_a + detail::UIndex(1)),
                detail::reshape_impl(b, detail::UIndex(1) + shape_b), Axes(n),
                Axes(0));
  ab.transpose(to_public(axes_trans));
  return detail::reshape_impl(ab, shape_c);
};
```

(`ab.transpose(...)` is the lazy member transpose, as the old `transpose(T, axes)` call was.)

`tensordot(a, b, axes_a, axes_b)`: first statements

```cpp
  const detail::UIndex ax_a = normalize_axes(axes_a, a.rank());
  const detail::UIndex ax_b = normalize_axes(axes_b, b.rank());
```

use them instead of `axes_a`/`axes_b`; `Shape shape_a = a.shape(); Shape shape_b = b.shape(); Shape shape_c; Axes trans_axes_a; Axes trans_axes_b;` → `detail::UIndex` with `internal_shape()`; `Tensor<MatrixType> c(comm, shape_c, rank_row_c);` → `Tensor<MatrixType> c(comm, shape_c, rank_row_c, detail::internal);`; both `transpose(...)` calls → `detail::transpose_impl(...)`.

- [ ] **Step 9: Convert svd, psvd, qr, eigh, eig, solve**

For every function below whose parameters include `const Axes &axes_row` / `axes_col` (and `axes_row_a`, `axes_col_a`, `axes_row_b`, `axes_col_b`), replace the `assert(...size() > 0)` / `assert(debug::check_svd_axes(...))` head and the `Axes axes = axes_row + axes_col;` line by:

```cpp
  const detail::UIndex row = normalize_axes(axes_row, a.rank());
  const detail::UIndex col = normalize_axes(axes_col, a.rank());
  assert(row.size() > 0);
  assert(col.size() > 0);
  assert(debug::check_svd_axes(row, col, a.rank()));
  const detail::UIndex axes = row + col;
```

(for the generalized `eigh` and for `solve`, do the same with suffixes `_a` using `a.rank()` and `_b` using `b.rank()`; keep each function's existing set of asserts, applied to the normalized values; `solve` keeps `assert(col_b.size() >= 0)` as is), set `urank = row.size()` (or `rank_row_a = row_a.size()` etc.) instead of `axes_row.size()`, and then apply the conversion rules to the rest of the body:

| Function | Specific edits |
|---|---|
| `svd(a, axes_row, axes_col, s)` | `transpose(a, axes, urank)` → `detail::transpose_impl(...)`; `const Shape &shape = a_t.shape();` → `const detail::UIndex &shape = a_t.internal_shape();` |
| `svd(a, axes_row, axes_col, u, s, vt)` | same as above; `Shape shape_u; Shape shape_vt;` → `detail::UIndex`; both `Tensor<MatrixType>(a.get_comm(), shape_X, urank_X)` → add `, detail::internal` |
| `svd(a, s)`, `svd(a, u, s, vt)`, `psvd` (all four), `qr(a, q, r)`, `solve(a, vector b, x)`, `solve(a, b, x)` | unchanged (they call public functions with literal non-negative axes) |
| `qr(a, axes_row, axes_col, q, r)` | `const Shape shape_a = a.shape();` → `const detail::UIndex shape_a = a.internal_shape();`; `Shape shape;`, `Shape shape_q;`, `Shape shape_r;` → `detail::UIndex`; `reshape(transpose(a, axes, urank), Shape(d_row, d_col))` → `detail::reshape_impl(detail::transpose_impl(a, axes, urank), detail::UIndex(d_row, d_col))`; `Tensor<MatrixType> mat_r(mat_q.get_comm(), mat_q.shape(), 1);` → `(mat_q.get_comm(), mat_q.internal_shape(), 1, detail::internal)`; every `reshape(X, shape_q)` / `reshape(X, shape_r)` → `detail::reshape_impl(X, shape_q)` / `(X, shape_r)`; the `slice(mat_r, 0, 0, size)` / `slice(mat_q, 1, 0, size)` calls stay public |
| `eigh(a, w, z)`, `eigh(a, w)` | `Shape shape = a.shape();` → `const detail::UIndex shape = a.internal_shape();`; `transpose(a, Axes(0, 1), 1)` → `detail::transpose_impl(a, detail::UIndex(0, 1), 1)`; `Tensor<MatrixType>(a.get_comm(), Shape(n, n), 1)` → `(a.get_comm(), detail::UIndex(n, n), 1, detail::internal)` |
| `eigh(a, axes_row, axes_col, w, z)`, `eigh(a, axes_row, axes_col, w)` | `transpose` → `detail::transpose_impl`; `const Shape &shape = a_t.shape();` → `const detail::UIndex &shape = a_t.internal_shape();`; `Shape shape_z;` → `detail::UIndex`; `Tensor<MatrixType>(a.get_comm(), shape_z, urank)` → add `, detail::internal` |
| `eigh(a, axes_row_a, axes_col_a, b, axes_row_b, axes_col_b, w, z)` | both `transpose` → `detail::transpose_impl`; `const Shape &shape_a/_b = X_t.shape();` → `const detail::UIndex & ... internal_shape()`; `Shape shape_z;` → `detail::UIndex`; tagged constructor for `z` |
| `eig(a, w, z)`, `eig(a, w)` | `Shape shape = a.shape();` → `const detail::UIndex shape = a.internal_shape();`; `transpose(a, Axes(0, 1), 1).gather()` → `detail::transpose_impl(a, detail::UIndex(0, 1), 1).gather()`; `z_t(a.get_comm(), Shape(n, n), 1)` → `z_t(a.get_comm(), detail::UIndex(n, n), 1, detail::internal)` |
| `eig(a, axes_row, axes_col, w, z)`, `eig(a, axes_row, axes_col, w)` | `transpose(a, axes, urank).gather()` → `detail::transpose_impl(a, axes, urank).gather()`; `const Shape &shape = a_t.shape();` → `const detail::UIndex &shape = a_t.internal_shape();`; `Shape shape_z;` → `detail::UIndex`; `z_t(a.get_comm(), shape_z, urank)` → add `, detail::internal` |
| `solve(a, b, x, axes_row_a, axes_col_a, axes_row_b, axes_col_b)` | `transpose` ×2 → `detail::transpose_impl`; `const Shape &shape_a/_b = X_t.shape();` → `const detail::UIndex & ... internal_shape()`; `Shape shape_x;` → `detail::UIndex`; `x = reshape(b_t, shape_x);` → `x = detail::reshape_impl(b_t, shape_x);` |

Also convert `operator<<(std::ostream&, const Tensor&)` at the end of `tensor_impl.hpp`: `Shape shape = t.shape();` → `const detail::UIndex &shape = t.internal_shape();`; `std::vector<std::size_t> idx(dim);` → `Index idx; idx.resize(dim);`; and `idx[d] = (i % accum[d]) / accum[d + 1];` → `idx[d] = static_cast<std::ptrdiff_t>((i % accum[d]) / accum[d + 1]);`; the comparisons `idx[dim - 1] == 0` and `idx[d - 1] == shape[d - 1] - 1` become `idx[dim - 1] == 0` and `static_cast<size_t>(idx[d - 1]) == shape[d - 1] - 1` (also the `idx[dim - 1] == shape[dim - 1] - 1` check).

- [ ] **Step 10: Convert `rsvd_impl.hpp` and `file_io/load.hpp`**

`rsvd(a, axes_row, axes_col, u, s, vt, target_rank, oversamp)`: apply the same head as Step 9 (`row`, `col`, `axes`), `rank_row = row.size()`, `rank_col = col.size()`; `transpose(a, axes, rank_row)` → `detail::transpose_impl(a, axes, rank_row)`; `const Shape &shape = a_t.shape();` → `const detail::UIndex &shape = a_t.internal_shape();`; `Shape shape_omega;` → `detail::UIndex shape_omega;`; `Tensor<MatrixType> omega(a.get_comm(), shape_omega, rank_col);` → add `, detail::internal`. The calls with `range(...)` and `Axes(rank_row)` stay public.

`rsvd(multiply_row, multiply_col, shape_row, shape_col, u, s, vt, target_rank, oversamp)`: `Shape shape_omega = shape_col;` → `detail::UIndex shape_omega = to_internal_shape(shape_col);` and add `to_internal_shape(shape_row);` as a discarded validation call (`(void)to_internal_shape(shape_row);`) at the top; `Tensor<MatrixType> omega(u.get_comm(), shape_omega, rank_col);` → add `, detail::internal`.

`file_io/load.hpp`: `Shape loaded_shape; Axes loaded_map;` and `Shape shape; Axes map;` → `detail::UIndex`.

- [ ] **Step 11: Build and fix remaining compile errors**

Run: `cmake --build build/si-mpi -j2 2>&1 | grep -E "error" | head -30`
Fix each remaining error by applying the conversion rules table (it is complete for the code paths listed above; any other error is a site the table already covers, e.g. a local `Shape`/`Index` that holds internal values). **Do not change any file under `tests/`** except `negative_index.cc`, `tests/index/*`, and the two `CMakeLists.txt` lines from Steps 2 and Task 1/2. If an existing test fails to compile, stop and report — it signals a breaking change not anticipated by spec Section 5.5.
Expected after fixing: build succeeds with no errors.

- [ ] **Step 12: Run the new tests**

Run: `for n in 1 2 3 4; do mpiexec -n $n build/si-mpi/tests/tensor/test_tensor --gtest_filter='NegativeIndex*' --gtest_brief=1 2>&1 | tail -2; done`
Expected: `[  PASSED  ] 22 tests.` for each process count (11 tests × 2 types).

- [ ] **Step 13: Run the whole suite in both builds**

Run:
```bash
ctest --test-dir build/si-mpi --output-on-failure
cmake --build build/si-nompi -j2 && ctest --test-dir build/si-nompi --output-on-failure
git diff --stat HEAD -- tests/tensor tests/file_io tests/common tests/legacy
```
Expected: all tests pass (17 MPI, 5 serial); the `git diff` shows only `tests/tensor/CMakeLists.txt` and the new `tests/tensor/negative_index.cc`.

- [ ] **Step 14: Commit**

```bash
git add include src tests/index tests/tensor
git commit -m "Accept negative axes and indices by making Index signed

Index, Axes, and Shape now hold std::ptrdiff_t. Public functions normalize
their arguments once at the entry point and throw std::out_of_range or
std::invalid_argument before any communication; the library works on
size_t internally.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Warnings, legacy, examples, Release

**Files:** possibly `include/mptensor/*.hpp` (warning fixes only)

- [ ] **Step 1: Warning check on new and modified code**

Run:
```bash
cmake -S . -B build/si-warn -G Ninja -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wsign-compare" > /dev/null
cmake --build build/si-warn -j2 2>&1 | grep -E "warning:" | grep -E "include/mptensor/(index|tensor|tensor_impl|rsvd_impl)\.hpp|file_io/load\.hpp|src/tensor\.cc|tests/(index|tensor/negative_index)" | sort -u
```
Expected: no output. For each warning printed, fix it in the named file (typically a signed/unsigned comparison: cast the signed side with `static_cast<size_t>` after it is known to be non-negative), rebuild, and re-run the command. Warnings in files this branch did not touch are out of scope; list them in the commit message if any appear in touched files only because of the type change.

- [ ] **Step 2: Legacy tests**

Run:
```bash
cmake -S . -B build/si-mpi -DBUILD_LEGACY_TESTS=ON > /dev/null && cmake --build build/si-mpi -j2 2>&1 | grep -E "error" | head
ctest --test-dir build/si-mpi --output-on-failure -R legacy/
```
Expected: builds; all 10 `legacy/` tests pass. If legacy fails to compile, stop and report (spec 7.1: legacy is frozen).

- [ ] **Step 3: Examples**

Run: `ninja -C build/si-mpi examples/all && ninja -C build/si-nompi -k 0 examples/all 2>&1 | grep -E "^FAILED"`
Expected: MPI build succeeds; without MPI, only `examples/save_load/...` and `examples/benchmark/.../rsvd...` fail (pre-existing, spec 7.5).

- [ ] **Step 4: Release builds**

Run:
```bash
for o in "" "-DENABLE_MPI=OFF"; do d=build/si-rel$o; cmake -S . -B "$d" -G Ninja -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release $o > /dev/null && cmake --build "$d" -j2 > /dev/null && ctest --test-dir "$d" --output-on-failure | tail -2; done
```
Expected: 100% tests passed in both.

- [ ] **Step 5: Commit (only if Step 1 required fixes)**

```bash
git add include src tests/index tests/tensor
git commit -m "Fix sign-compare warnings introduced by the signed Index

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Clean up**

Run: `rm -rf build/si-warn "build/si-rel" "build/si-rel-DENABLE_MPI=OFF"`
