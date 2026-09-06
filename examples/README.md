# Examples

Small programs that use the library the way a stranger would: through
`find_package(pnl)` against an installed prefix, from a project that declares
`LANGUAGES CXX` and nothing else.

```bash
make install-test          # from the repository root: install, build, run
```

That target installs into `build/stage`, configures this directory against it
with `-DCMAKE_PREFIX_PATH=build/stage`, builds it and runs both programs. It is
the phase B1 gate and continuous integration runs it too, so an export that
breaks for a consumer breaks the build rather than being discovered by one.

To do it by hand against a prefix of your own:

```bash
cmake --install build --prefix /some/prefix
cmake -S examples -B examples/build -DCMAKE_PREFIX_PATH=/some/prefix
cmake --build examples/build
./examples/build/poisson
./examples/build/custom_backend
```

| Example | What it shows |
| --- | --- |
| `poisson.cpp` | A 2D Poisson problem solved with conjugate gradient on the OpenMP backend where the install has one and the serial backend otherwise, printing the iteration count, the relative residual and the library version. |
| `custom_backend.cpp` | An execution backend of the reader's own, registered against the installed package with one namespace scope object, selected by name through `make_backend`, and used to run conjugate gradient over a Poisson problem. It prints that its iterate is bit identical to the serial backend's, every value, compared with `==`. |

Both link `pnl::core` and nothing else, which is the claim: an extension point
that needs a private header or a build flag is not an extension point.
