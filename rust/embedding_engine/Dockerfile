FROM ghcr.io/pyo3/manylinux_2_28_aarch64:latest

# Use bundled Python 3.12 in /opt/python/cp312-cp312
ENV PYO3_PYTHON=/opt/python/cp312-cp312/bin/python \
    PATH="/root/.cargo/bin:/opt/python/cp312-cp312/bin:${PATH}"

WORKDIR /workspace
CMD ["bash"]
