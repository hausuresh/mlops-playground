SHELL := /bin/bash
VERSION := 1.15.0-dev
KIND_NAME ?= kind
IMAGE_NAME_BASE=mlflowservercustom
IMAGE_NAME=haunv/${IMAGE_NAME_BASE}
IMAGE_VERSION=latest

build:
	s2i build \
		-E environment \
		./mlflowserver \
		seldonio/seldon-core-s2i-python37-ubi8:${VERSION} \
		${IMAGE_NAME}:${IMAGE_VERSION}

push:
	docker push ${IMAGE_NAME}:${VERSION}

kind_load: build
	kind load -v 3 docker-image ${IMAGE_NAME}:${VERSION} --name ${KIND_NAME}