IMAGE_NAME=dbt-vectors-embed

.PHONY: docker-build docker-test

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-test: docker-build
	docker run --rm \
	  -e PYO3_PYTHON=/usr/local/bin/python3 \
	  -e PYO3_LIB_DIR=/usr/local/lib \
	  -e PYO3_INCLUDE_DIR=/usr/local/include/python3.12 \
	  -v $(PWD):/workspace -w /workspace $(IMAGE_NAME) \
	  sh -c "rm -rf target && cargo test -q"
