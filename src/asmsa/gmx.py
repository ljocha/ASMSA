import kubernetes as k8s
import tempfile
import yaml
import uuid
import os
import time

class GMX:
	def __init__(self,pvc=None,image='ljocha/gromacs:2023-1'):
		self.pvc = pvc
		self.name = None
		self.ns = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read()
		self.image = image
		k8s.config.load_incluster_config()
		self.batchapi = k8s.client.BatchV1Api()
		self.coreapi = k8s.client.CoreV1Api()

	def start(self,cmd,input=None,gpus=0,cores=1,mem=4,wait=False,delete=False,tail=10):
		
		if self.name:
			raise RuntimeError(f"job {self.name} already running, delete() it first")
			
		self.name = "gmx-" + str(uuid.uuid4())
		if isinstance(cmd,list):
			cmd = ' '.join(map(lambda s: f'"{s}"',cmd))

		if input is not None:
			cmd += f' <<<"{input}"'
		
		kcmd = ['bash', '-c', 'gmx ' + cmd]
		
		yml = f"""\
apiVersion: batch/v1
kind: Job
metadata:
  name: {self.name}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: {self.name}
        image: {self.image}
        workingDir: /mnt/ASMSA
        command: {kcmd}
        securityContext:
          runAsUser: 1000
          runAsGroup: 1000
        env:
        - name: 'OMP_NUM_THREADS'
          value: '{cores}'
        resources:
          requests:
            cpu: '{cores}'
            memory: {mem}Gi
            nvidia.com/gpu: {gpus}
          limits:
            cpu: '{cores}'
            memory: {mem}Gi
            nvidia.com/gpu: {gpus}
        volumeMounts:
        - name: vol-1
          mountPath: /mnt
      volumes:
      - name: vol-1
        persistentVolumeClaim:
          claimName: {self.pvc}
"""
				
		yml = yaml.safe_load(yml)
		self.job = self.batchapi.create_namespaced_job(self.ns,yml)

		if wait:
#			print(self.status().succeeded)
			while not self.status().succeeded:
#				print(self.status().succeeded)
				print('.',end='')
				time.sleep(2)
			print()

			self.log(tail=tail)
			if delete:
				self.delete()
				
				
	def status(self,pretty=True):
		if self.name:
			return self.batchapi.read_namespaced_job(self.name, self.ns, pretty=pretty).status
		return None
		
	def delete(self):
		if self.name:
			self.batchapi.delete_namespaced_job(self.name, self.ns)
			self.name = None
		return None
		
	def log(self, tail=None):
		if self.name:
			if tail:
				os.system(f"kubectl logs job/{self.name} | tail -{tail}")
			else:
				os.system(f"kubectl logs job/{self.name}")
		return None
