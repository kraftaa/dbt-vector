from setuptools import Distribution, setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except Exception:
        _bdist_wheel = None


if _bdist_wheel is not None:
    class BinaryDistribution(Distribution):
        # Ensure wheel is treated as platform-specific (platlib), not purelib.
        def has_ext_modules(self):
            return True


    class bdist_wheel(_bdist_wheel):
        # Force platform wheels because we bundle a native Rust binary.
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        # The package works with any Python 3 version; only the platform matters.
        def get_tag(self):
            _py, _abi, plat = super().get_tag()
            return "py3", "none", plat


    setup(cmdclass={"bdist_wheel": bdist_wheel}, distclass=BinaryDistribution)
else:
    setup()
