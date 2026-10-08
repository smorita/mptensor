# Tensor Single Template Parameter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `Tensor<Matrix, C>` (two params: template template + scalar) to `Tensor<MatrixType>` (one param: fully-instantiated matrix type), e.g. `Tensor<lapack::Matrix<double>>`.

**Architecture:** Add a `rebind<D>` type alias to each Matrix class so that "change scalar type" operations remain possible. Then mechanically substitute the template parameter everywhere: `Matrix<C>` → `MatrixType`, `C` → `value_type` (or `typename MatrixType::value_type`), `Tensor<Matrix, C>` → `Tensor<MatrixType>`. Special cases (`Tensor<Matrix, complex>`, `Tensor<lapack::Matrix, C>`) use `rebind` and direct instantiation respectively.

**Tech Stack:** C++17, Intel icpx, CMake/Ninja, MPI (ScaLAPACK) + no-MPI (LAPACK) builds.

---

## Background

### Current pattern

```cpp
template <template <typename> class Matrix, typename C>
class Tensor { ... };

// Usage
Tensor<lapack::Matrix, double> t;
```

### Target pattern

```cpp
template <typename MatrixType>
class Tensor { ... };

// Usage
Tensor<lapack::Matrix<double>> t;
```

### Key substitution rules

| Before | After |
|--------|-------|
| `template <template <typename> class Matrix, typename C>` | `template <typename MatrixType>` |
| `template <template <typename> class Matrix>` (scalar-specialized) | merged into single template (use `if constexpr`) |
| `Matrix<C>` | `MatrixType` |
| `Tensor<Matrix, C>` | `Tensor<MatrixType>` |
| `Tensor<lapack::Matrix, C>` | `Tensor<lapack::Matrix<value_type>>` |
| `Tensor<Matrix, complex>` | `Tensor<typename MatrixType::template rebind<complex>>` |
| `Tensor<lapack::Matrix, complex>` | `Tensor<lapack::Matrix<complex>>` |
| Free return type `C` | `typename MatrixType::value_type` |
| `Matrix<C>::matrix_type_tag` | `MatrixType::matrix_type_tag` |

### Files to modify

| File | What changes |
|------|-------------|
| `include/mptensor/matrix/lapack/matrix_lapack.hpp` | Add `rebind` alias |
| `include/mptensor/matrix/scalapack/matrix_scalapack.hpp` | Add `rebind` alias |
| `include/mptensor/tensor.hpp` | Class decl + all free function decls |
| `include/mptensor/tensor_impl.hpp` | All member + free function impls (~2685 lines) |
| `include/mptensor/rsvd.hpp` | Free function decls |
| `include/mptensor/rsvd_impl.hpp` | Free function impls |
| `include/mptensor/file_io/save.hpp` | `save` / `save_ver_0_2` impls |
| `include/mptensor/file_io/load.hpp` | `load` / `load_ver_0_2` impls |
| `include/mptensor/mptensor.hpp` | `DTensor`/`ZTensor` aliases |
| `tests/legacy/typedef.hpp` | `TensorD`/`TensorC` typedefs |

---

## Task 1: Add `rebind` type alias to Matrix classes

**Files:**
- Modify: `include/mptensor/matrix/lapack/matrix_lapack.hpp` (inside `class Matrix<C>` public section)
- Modify: `include/mptensor/matrix/scalapack/matrix_scalapack.hpp` (inside `class Matrix<C>` public section)

This alias is needed so that `eig()` can produce `Tensor<typename MatrixType::template rebind<complex>>` without knowing the concrete matrix template.

- [ ] **Step 1: Add `rebind` to `lapack::Matrix`**

In `include/mptensor/matrix/lapack/matrix_lapack.hpp`, add inside the `public:` section of `class Matrix` (e.g. right after `using comm_type  = int;`):

```cpp
  template <typename D>
  using rebind = Matrix<D>;
```

- [ ] **Step 2: Add `rebind` to `scalapack::Matrix`**

In `include/mptensor/matrix/scalapack/matrix_scalapack.hpp`, add inside the `public:` section of `class Matrix` (e.g. right after `using comm_type  = MPI_Comm;`):

```cpp
  template <typename D>
  using rebind = Matrix<D>;
```

- [ ] **Step 3: Verify compilation of matrix headers alone**

```bash
cd /home/morita/project/mptensor/build
ninja -j4 2>&1 | head -30
```

Expected: same output as before (headers are included by tensor, not compiled standalone).

---

## Task 2: Update `tensor.hpp` — class declaration

**Files:**
- Modify: `include/mptensor/tensor.hpp`

This file contains:
1. The `Tensor` class template declaration (lines 53–174)
2. ~50 free function declarations (lines 176–341)

- [ ] **Step 1: Change class template parameter**

Replace:
```cpp
template <template <typename> class Matrix, typename C>
class Tensor {
 public:
  using value_type  = C;                              //!< \c double or \c complex
  using matrix_type = Matrix<C>;                      //!< type of Matrix class
  using comm_type   = typename Matrix<C>::comm_type;  //!< type of communicator. \c MPI_Comm or \c int.
```

With:
```cpp
template <typename MatrixType>
class Tensor {
 public:
  using value_type  = typename MatrixType::value_type;  //!< \c double or \c complex
  using matrix_type = MatrixType;                       //!< type of Matrix class
  using comm_type   = typename MatrixType::comm_type;   //!< type of communicator. \c MPI_Comm or \c int.
```

- [ ] **Step 2: Update constructor taking lapack tensor**

Replace:
```cpp
  Tensor(const comm_type &, const Tensor<lapack::Matrix, C> &);
```
With:
```cpp
  Tensor(const comm_type &, const Tensor<lapack::Matrix<value_type>> &);
```

- [ ] **Step 3: Update `get_matrix` return types and `Mat` storage**

Replace:
```cpp
  const Matrix<C> &get_matrix() const;
  Matrix<C> &get_matrix();
```
With:
```cpp
  const MatrixType &get_matrix() const;
  MatrixType &get_matrix();
```

Replace the private member:
```cpp
  Matrix<C> Mat;  //!< local storage.
```
With:
```cpp
  MatrixType Mat;  //!< local storage.
```

- [ ] **Step 4: Update all member function return types inside the class**

All occurrences of `Tensor<Matrix, C>` in member function signatures inside the class body → `Tensor<MatrixType>`.

`gather()` return type:
```cpp
  // Before:
  Tensor<lapack::Matrix, C> gather();
  // After:
  Tensor<lapack::Matrix<value_type>> gather();
```

- [ ] **Step 5: Update all free function declarations**

Every `template <template <typename> class Matrix, typename C>` before a free function → `template <typename MatrixType>`.

Every `Tensor<Matrix, C>` in free function signatures → `Tensor<MatrixType>`.

For `eig` declarations returning `Tensor<Matrix, complex>`:
```cpp
// Before:
template <template <typename> class Matrix, typename C>
int eig(const Tensor<Matrix, C> &a, std::vector<complex> &eigval,
        Tensor<Matrix, complex> &eigvec);

// After:
template <typename MatrixType>
int eig(const Tensor<MatrixType> &a, std::vector<complex> &eigval,
        Tensor<typename MatrixType::template rebind<complex>> &eigvec);
```

For `operator*` that also has `typename D`:
```cpp
// Before:
template <template <typename> class Matrix, typename C, typename D>
Tensor<Matrix, C> operator*(Tensor<Matrix, C> lhs, D rhs);

// After:
template <typename MatrixType, typename D>
Tensor<MatrixType> operator*(Tensor<MatrixType> lhs, D rhs);
```

- [ ] **Step 6: Update `sqrt` and `conj` declarations**

```cpp
// Before (two specializations):
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> sqrt(Tensor<Matrix, C> t);
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> conj(Tensor<Matrix, C> t);

// After (single declaration):
template <typename MatrixType>
Tensor<MatrixType> sqrt(Tensor<MatrixType> t);
template <typename MatrixType>
Tensor<MatrixType> conj(Tensor<MatrixType> t);
```

- [ ] **Step 7: Attempt build to see remaining errors**

```bash
cd /home/morita/project/mptensor/build
ninja -j4 2>&1 | grep "error:" | head -40
```

Expected: errors in `tensor_impl.hpp`, `rsvd.hpp`, `rsvd_impl.hpp` (not yet updated).

---

## Task 3: Update `tensor_impl.hpp` — member and free function implementations

**Files:**
- Modify: `include/mptensor/tensor_impl.hpp` (~2685 lines)

This is the largest file. Work through it section by section. The pattern is mechanical: replace template headers and type expressions per the substitution table.

- [ ] **Step 1: Replace all template headers**

Every occurrence of:
```cpp
template <template <typename> class Matrix, typename C>
```
Replace with:
```cpp
template <typename MatrixType>
```

There are also specialized templates (used for `sqrt`/`conj`):
```cpp
template <template <typename> class Matrix>
```
These will be handled separately in Step 5.

Run: `grep -n "template <template" include/mptensor/tensor_impl.hpp` to get all line numbers.

- [ ] **Step 2: Replace member function qualifiers**

Every:
```cpp
Tensor<Matrix, C>::
```
→ `Tensor<MatrixType>::`

- [ ] **Step 3: Replace `Tensor<Matrix, C>` in return types and parameters**

Every `Tensor<Matrix, C>` → `Tensor<MatrixType>`.

- [ ] **Step 4: Replace `Tensor<lapack::Matrix, C>` occurrences**

Occurrences appear in:
- `gather()` implementation (line ~955): return type and local variable
- `eig()` implementations (lines ~2301, ~2330, ~2372, ~2421): intermediate computations

Replace all `Tensor<lapack::Matrix, C>` → `Tensor<lapack::Matrix<value_type>>`.

Inside member functions, `value_type` is directly accessible. Inside free functions, add `using value_type = typename MatrixType::value_type;` at the top of each function body where needed.

Example for `gather()`:
```cpp
// Before:
template <template <typename> class Matrix, typename C>
Tensor<lapack::Matrix, C> Tensor<Matrix, C>::gather() {
  ...
  Tensor<lapack::Matrix, C> T(get_comm(), get_matrix().flatten());
  ...
}

// After:
template <typename MatrixType>
Tensor<lapack::Matrix<value_type>> Tensor<MatrixType>::gather() {
  ...
  Tensor<lapack::Matrix<value_type>> T(get_comm(), get_matrix().flatten());
  ...
}
```

- [ ] **Step 5: Replace `Tensor<Matrix, complex>` and `Tensor<lapack::Matrix, complex>` in `eig` implementations**

```cpp
// Before (in eig for general matrix):
Tensor<lapack::Matrix, complex> z_t(a.get_comm(), Shape(n, n), 1);
z = Tensor<Matrix, complex>(a.get_comm(), z_t);

// After:
Tensor<lapack::Matrix<complex>> z_t(a.get_comm(), Shape(n, n), 1);
z = Tensor<typename MatrixType::template rebind<complex>>(a.get_comm(), z_t);
```

The `eig` free function signatures also change (matching Task 2 Step 5):
```cpp
// Before:
template <template <typename> class Matrix, typename C>
int eig(const Tensor<Matrix, C> &a, std::vector<complex> &w,
        Tensor<Matrix, complex> &z) {

// After:
template <typename MatrixType>
int eig(const Tensor<MatrixType> &a, std::vector<complex> &w,
        Tensor<typename MatrixType::template rebind<complex>> &z) {
  using value_type = typename MatrixType::value_type;
```

- [ ] **Step 6: Replace free function return types that were `C`**

Examples like `trace`, `max`, `min`:
```cpp
// Before:
template <template <typename> class Matrix, typename C>
C trace(const Tensor<Matrix, C> &a) {

// After:
template <typename MatrixType>
typename MatrixType::value_type trace(const Tensor<MatrixType> &a) {
  using value_type = typename MatrixType::value_type;
```

In function bodies where `C val;` or `C result;` etc. appear, replace `C` with `value_type` (after adding the `using` alias).

- [ ] **Step 7: Update `sqrt` and `conj` implementations**

The old code has four separate template specializations. Replace with two `if constexpr` functions:

```cpp
//! \cond
template <typename MatrixType>
inline Tensor<MatrixType> sqrt(Tensor<MatrixType> t) {
  using C = typename MatrixType::value_type;
  if constexpr (std::is_same_v<C, double>) {
    return t.map(static_cast<double (*)(double)>(&std::sqrt));
  } else {
    return t.map(static_cast<complex (*)(const complex &)>(&std::sqrt));
  }
}
template <typename MatrixType>
inline Tensor<MatrixType> conj(Tensor<MatrixType> t) {
  using C = typename MatrixType::value_type;
  if constexpr (std::is_same_v<C, double>) {
    return t;
  } else {
    return t.map(static_cast<complex (*)(const complex &)>(&std::conj));
  }
}
//! \endcond
```

Note: Add `#include <type_traits>` if not already present.

- [ ] **Step 8: Replace `Matrix<C>::matrix_type_tag` and similar static accesses**

In `save.hpp` / `load.hpp`:
```cpp
// Before:
Matrix<C>::matrix_type_tag
Matrix<C>::matrix_type_name

// After:
MatrixType::matrix_type_tag
MatrixType::matrix_type_name
```

- [ ] **Step 9: Replace `sizeof(C)` occurrences**

In `save.hpp` and `load.hpp` where binary I/O uses `sizeof(C)`:
```cpp
// Before:
sizeof(C) * local_size()

// After:
sizeof(value_type) * local_size()
```

Inside member functions, `value_type` is accessible directly. Add `using value_type = typename MatrixType::value_type;` inside free function bodies if needed.

- [ ] **Step 10: Attempt partial build**

```bash
cd /home/morita/project/mptensor/build
ninja -j4 2>&1 | grep "error:" | head -40
```

Expected: errors only in `rsvd.hpp`/`rsvd_impl.hpp` and remaining unchanged files.

---

## Task 4: Update `rsvd.hpp` and `rsvd_impl.hpp`

**Files:**
- Modify: `include/mptensor/rsvd.hpp`
- Modify: `include/mptensor/rsvd_impl.hpp`

- [ ] **Step 1: Update `rsvd.hpp` declarations**

Replace all `template <template <typename> class Matrix, typename C>` with `template <typename MatrixType>`.

For the variants with extra `Func1`, `Func2` parameters:
```cpp
// Before:
template <template <typename> class Matrix, typename C, typename Func1, typename Func2>
int rsvd(Func1 &multiply_row, Func2 &multiply_col, const Shape &shape_row,
         const Shape &shape_col, Tensor<Matrix, C> &u, std::vector<double> &s,
         Tensor<Matrix, C> &vt, const size_t target_rank, const size_t oversamp);

// After:
template <typename MatrixType, typename Func1, typename Func2>
int rsvd(Func1 &multiply_row, Func2 &multiply_col, const Shape &shape_row,
         const Shape &shape_col, Tensor<MatrixType> &u, std::vector<double> &s,
         Tensor<MatrixType> &vt, const size_t target_rank, const size_t oversamp);
```

Replace all `Tensor<Matrix, C>` → `Tensor<MatrixType>`.

- [ ] **Step 2: Update `rsvd_impl.hpp` implementations**

Same pattern:
- `template <template <typename> class Matrix, typename C>` → `template <typename MatrixType>`
- `template <template <typename> class Matrix, typename C, typename Func1, typename Func2>` → `template <typename MatrixType, typename Func1, typename Func2>`
- `Tensor<Matrix, C>` → `Tensor<MatrixType>`

Local variables of type `Tensor<Matrix, C>`:
```cpp
// Before:
Tensor<Matrix, C> a_t = ...;
Tensor<Matrix, C> omega(...);
Tensor<Matrix, C> r;
Tensor<Matrix, C> q;

// After:
Tensor<MatrixType> a_t = ...;
Tensor<MatrixType> omega(...);
Tensor<MatrixType> r;
Tensor<MatrixType> q;
```

- [ ] **Step 3: Build and check**

```bash
cd /home/morita/project/mptensor/build
ninja -j4 2>&1 | grep "error:" | head -40
```

Expected: errors only in `save.hpp`/`load.hpp`.

---

## Task 5: Update `file_io/save.hpp` and `file_io/load.hpp`

**Files:**
- Modify: `include/mptensor/file_io/save.hpp`
- Modify: `include/mptensor/file_io/load.hpp`

- [ ] **Step 1: Update `save.hpp`**

Replace template headers:
```cpp
// Before:
template <template <typename> class Matrix, typename C>
void Tensor<Matrix, C>::save(const std::string &filename) const {
  ...
  fout << "matrix_type= " << Matrix<C>::matrix_type_tag;
  fout << " (" << Matrix<C>::matrix_type_name << ")\n";
  fout << "value_type= " << value_type_tag<C>();
  fout << " (" << value_type_name<C>() << ")\n";
  ...
  sizeof(C) * local_size()
  ...
}

// After:
template <typename MatrixType>
void Tensor<MatrixType>::save(const std::string &filename) const {
  ...
  fout << "matrix_type= " << MatrixType::matrix_type_tag;
  fout << " (" << MatrixType::matrix_type_name << ")\n";
  fout << "value_type= " << value_type_tag<value_type>();
  fout << " (" << value_type_name<value_type>() << ")\n";
  ...
  sizeof(value_type) * local_size()
  ...
}
```

Also update `save_ver_0_2`:
```cpp
// Before:
template <template <typename> class Matrix, typename C>
void Tensor<Matrix, C>::save_ver_0_2(const char *filename) const {
  ...
  sizeof(C) * local_size()

// After:
template <typename MatrixType>
void Tensor<MatrixType>::save_ver_0_2(const char *filename) const {
  ...
  sizeof(value_type) * local_size()
```

- [ ] **Step 2: Update `load.hpp`**

Read the file first and apply the same pattern:
- `template <template <typename> class Matrix, typename C>` → `template <typename MatrixType>`
- `Tensor<Matrix, C>::` → `Tensor<MatrixType>::`
- `sizeof(C)` → `sizeof(value_type)`
- Any `Matrix<C>::` static access → `MatrixType::`

- [ ] **Step 3: Build and check**

```bash
cd /home/morita/project/mptensor/build
ninja -j4 2>&1 | grep "error:" | head -40
```

Expected: clean build, or only linker/example errors.

---

## Task 6: Update type aliases and test typedefs

**Files:**
- Modify: `include/mptensor/mptensor.hpp`
- Modify: `tests/legacy/typedef.hpp`

- [ ] **Step 1: Update `mptensor.hpp`**

```cpp
// Before:
#ifdef _NO_MPI
using DTensor = Tensor<lapack::Matrix, double>;
using ZTensor = Tensor<lapack::Matrix, complex>;
#else
using DTensor = Tensor<scalapack::Matrix, double>;
using ZTensor = Tensor<scalapack::Matrix, complex>;
#endif

// After:
#ifdef _NO_MPI
using DTensor = Tensor<lapack::Matrix<double>>;
using ZTensor = Tensor<lapack::Matrix<complex>>;
#else
using DTensor = Tensor<scalapack::Matrix<double>>;
using ZTensor = Tensor<scalapack::Matrix<complex>>;
#endif
```

- [ ] **Step 2: Update `tests/legacy/typedef.hpp`**

```cpp
// Before:
#ifdef _NO_MPI
typedef Tensor<lapack::Matrix, double> TensorD;
typedef Tensor<lapack::Matrix, complex> TensorC;
#else
typedef Tensor<scalapack::Matrix, double> TensorD;
typedef Tensor<scalapack::Matrix, complex> TensorC;
#endif

// After:
#ifdef _NO_MPI
typedef Tensor<lapack::Matrix<double>> TensorD;
typedef Tensor<lapack::Matrix<complex>> TensorC;
#else
typedef Tensor<scalapack::Matrix<double>> TensorD;
typedef Tensor<scalapack::Matrix<complex>> TensorC;
#endif
```

- [ ] **Step 3: Check for any other user-facing `Tensor<X, Y>` usages in examples or tests**

```bash
grep -rn "Tensor<" /home/morita/project/mptensor/tests/ /home/morita/project/mptensor/examples/ \
  | grep -v "Tensor<MatrixType\|Tensor<lapack\|Tensor<scalapack\|TensorD\|TensorC\|#" \
  | head -30
```

Fix any remaining two-argument `Tensor<X, Y>` usages found.

- [ ] **Step 4: Full build**

```bash
cd /home/morita/project/mptensor/build
ninja -j4 2>&1 | tail -20
```

Expected: successful build with no errors.

---

## Task 7: Run tests and fix any remaining issues

**Files:** No file changes expected; fix issues as found.

- [ ] **Step 1: Run the test suite**

```bash
cd /home/morita/project/mptensor/build
ctest --output-on-failure 2>&1 | tail -40
```

- [ ] **Step 2: Fix any test failures**

If tests fail due to template instantiation issues, check:
1. Is the `rebind` alias properly in scope?
2. Are all `Tensor<Matrix, C>` occurrences replaced (use `grep -rn "Tensor<[A-Za-z]*," include/`)?
3. Are there any remaining `template <typename> class Matrix` patterns?

```bash
grep -rn "template <typename> class Matrix\|Tensor<[A-Za-z]*," \
  /home/morita/project/mptensor/include/ | grep -v "^Binary"
```

- [ ] **Step 3: Commit**

```bash
git add include/mptensor/matrix/lapack/matrix_lapack.hpp \
        include/mptensor/matrix/scalapack/matrix_scalapack.hpp \
        include/mptensor/tensor.hpp \
        include/mptensor/tensor_impl.hpp \
        include/mptensor/rsvd.hpp \
        include/mptensor/rsvd_impl.hpp \
        include/mptensor/file_io/save.hpp \
        include/mptensor/file_io/load.hpp \
        include/mptensor/mptensor.hpp \
        tests/legacy/typedef.hpp
git commit -m "Simplify Tensor template: <Matrix, C> -> <MatrixType>"
```

---

## Notes on tricky cases

### `eig` returning complex tensor from real input

The `eig()` functions take `Tensor<MatrixType>` (possibly `real`) and return `Tensor<typename MatrixType::template rebind<complex>>`. The `rebind` alias added in Task 1 makes this work:

```cpp
// lapack::Matrix<double>::rebind<complex> = lapack::Matrix<complex>
// scalapack::Matrix<double>::rebind<complex> = scalapack::Matrix<complex>
```

### `gather()` always returns lapack tensor

`gather()` collects distributed data into a non-distributed lapack tensor. Its return type is always `Tensor<lapack::Matrix<value_type>>` regardless of the input matrix backend. This is fine and more explicit than before.

### `sqrt`/`conj` scalar-type dispatch

The old code used two separate template specializations (one for `double`, one for `complex`) via `template <template <typename> class Matrix>`. The new code uses `if constexpr (std::is_same_v<value_type, double>)` inside a single template. Requires C++17.
