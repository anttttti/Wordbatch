IMAGE_NAME=wordbatch
CONTAINER_NAME=wordbatch_dev

build: ## Build the image
	docker build -t $(IMAGE_NAME) .

run-dev: ## Run container for development
	docker run \
		-it \
		--ipc=host \
		--name=$(CONTAINER_NAME) \
		-v $(shell pwd):/wordbatch $(IMAGE_NAME) bash
		
attach: ## Run a bash in a running container
	docker start $(CONTAINER_NAME) && docker attach $(CONTAINER_NAME)

stop: ## Stop and remove a running container
	docker stop $(CONTAINER_NAME); docker rm $(CONTAINER_NAME)
