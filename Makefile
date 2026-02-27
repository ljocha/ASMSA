
default:
	@echo Choose from install_local, run_local

install_local:
	mamba env create -y -n asmsa -f binder/environment.yml

run_local:
	mamba run -e PATH=${PWD}/miscellaneous/bin-local:${PATH} -n asmsa jupyter lab

