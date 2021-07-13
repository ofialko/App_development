project = $(shell basename $(abspath $(dir $$PWD)))
project_l = $(shell echo $(project) | tr A-Z a-z)

image = image_$(project_l)
container = container_$(project_l)

build:
	docker build -t $(image) .

run:
	docker run -v $$(pwd):/app -p 8000:8000 --rm --name $(container) $(image)

bash:
	docker exec -it $(container) /bin/bash
