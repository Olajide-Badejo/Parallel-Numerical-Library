# Examples

Small programs that use the library the way a stranger would: through
`find_package(pnl)` against an installed prefix, from a project that declares
`LANGUAGES CXX` and nothing else.

```bash
make install-test          # from the repository root: install, build, run
```

That target installs into `build/stage`, configures this directory against it
with `-DCMAKE_PREFIX_PATH=build/stage`, builds it and runs `poisson`. It is the
phase B1 gate and continuous integration runs it too, so an export that breaks
for a consumer breaks the build rather than being discovered by one.

To do it by hand against a prefix of your own:

```bash
cmake --install build --prefix /some/prefix
cmake -S examples -B examples/build -DCMAKE_PREFIX_PATH=/some/prefix
cmake --build examples/build
./examples/build/poisson
```

| Example | What it shows |
| --- | --- |
| `poisson.cpp` | A 2D Poisson problem solved with conjugate gradient on the OpenMP backend where the install has one and the serial backend otherwise, printing the iteration count, the relative residual and the library version. |

`custom_backend.cpp`, which registers a backend of the reader's own, arrives with
the registration hook in phase B4. There is nothing to register against yet, so
there is no stub for it here.
