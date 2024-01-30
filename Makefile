# Variables
PYTHON := python3
DOCKER := docker
PACKAGE_NAME := wham

# Targets
.PHONY: install demo docker-image

install:
	$(PIP) install .

docker-image:
	$(DOCKER) build . -t $(PACKAGE_NAME)

demo:
	@echo Running on file $1 in folder $0
	$(DOCKER) run -v $(directory):/input_data --rm --gpus all $(PACKAGE_NAME) bash /WHAM/run_demo.sh /input_data/$(video_name)