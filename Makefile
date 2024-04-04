image=$(shell cat IMAGE | cut -d: -f1)
tag=$(shell cat IMAGE | cut -d: -f2)

port?=9000

build: package-build docker-build

package-build:
	python3 -m build

docker-build: 
	docker build ${buildopt} -t ${image} .
	docker tag ${image}:latest ${image}:${tag}
	docker push ${image}:latest 
	docker push ${image}:${tag}

docker-amd:
	docker build -f Dockerfile.amd -t ${image}:amd .

gpus?=--gpus all
# flags=${gpus} --rm -u $(shell id -u) -w /work -v ${PWD}:/work -e HOME=/work  -e DOCKER_WORKDIR=${PWD} -v /var/run/docker.sock:/var/run/docker.sock --entrypoint /work/start_in_venv.sh
flags=${gpus} --rm -u $(shell id -u) --group-add $(shell grep docker /etc/group | cut -f3 -d:) -w /work -v ${PWD}:/work -e HOME=/work  -e DOCKER_WORKDIR=${PWD} -v /var/run/docker.sock:/var/run/docker.sock 
# flags=--rm -p ${port}:${port} -u $(shell id -u) -w /work -v ${PWD}:/work -e HOME=/work  --entrypoint /usr/bin/env

amdflags=--rm -u $(shell id -u) -w /work -v ${PWD}:/work -e HOME=/work  --entrypoint /work/start_in_venv.sh -p ${port}:${port} --device=/dev/kfd --device=/dev/dri --shm-size 16G --group-add video --group-add render 

lab notebook:
	docker run ${flags} -p ${port}:${port} ${image} jupyter-$@ --ip 0.0.0.0 --port ${port}

bash:
	docker run -ti ${flags} -p ${port}:${port} ${image} bash

nonet:
	docker run -ti ${flags} ${image} bash

amd:
	docker run -ti ${amdflags} ${image}:amd bash

