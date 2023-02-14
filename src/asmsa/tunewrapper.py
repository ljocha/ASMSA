import kubernetes as k8s
import socket
import yaml
import os
import dill


class TuneWrapper:
	def __init__(self, epochs=200, trials=80, hpfunc=None, port=8889, image=None, pvc=None, output='best.txt', pdb=None, xtc=None, top=None, ndx=None):
		self.epochs = epochs
		self.trials = trials
		self.hpfunc = hpfunc
		self.port = port
		self.slaves = 10
		if image is None:
			with open('IMAGE') as img:
				self.image = img.read().rstrip()
		else:
			self.image = image
		self.pvc = pvc
		self.output = output
		self.pdb = pdb
		self.xtc = xtc
		self.top = top
		self.ndx = ndx
		

	def master_start(self):
		with open('hpfunc.dill','wb') as d:
		    dill.dump(self.hpfunc,d)

		os.system(f"nohup tuning.py --output best.txt --pdb {self.pdb} --xtc {self.xtc} --top {self.top} --ndx {self.ndx} --master 0.0.0.0:{self.port} --id chief --epochs {self.epochs} --trials {self.trials} --hpfunc hpfunc.dill >master.out 2>master.err &")

	def master_status(self):
		with os.popen('ps axww | grep tuning.py | egrep -v "ps axww|grep"') as p,\
			open('master.out') as o:
				return p.read() + '\n' + o.read()



	def workers_start(self,num=None):
		master=socket.gethostbyname(os.environ['HOSTNAME'])
		slaves = num if num is not None else self.slaves


		job_template = """\
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
spec:
  backoffLimit: 0
  parallelism: {slaves}
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: {name}
        image: {image}
        workingDir: /mnt/ASMSA
        command: {command}
        securityContext:
          runAsUser: 1000
          runAsGroup: 1000
        env:
        - name: 'OMP_NUM_THREADS'
          value: '4'
        resources:
          requests:
            cpu: '4'
            memory: 4096Mi
          limits:
            cpu: '4'
            memory: 16384Mi
        volumeMounts:
        - name: vol-1
          mountPath: /mnt
      volumes:
      - name: vol-1
        persistentVolumeClaim:
          claimName: {pvc}
"""
		command=['/usr/local/bin/tuning.sh','--pdb',self.pdb,'--xtc',self.xtc,'--top',self.top,'--ndx',self.ndx,'--master',master+':'+str(self.port),'--epochs',str(self.epochs),'--hpfunc','hpfunc.dill']
		
		k8s.config.load_incluster_config()
		namespace = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read()
		job=job_template.format(name='tuner', image=self.image, command=command, pvc=self.pvc,slaves=slaves)
		batch_api = k8s.client.BatchV1Api()
		with open('job.yml','w') as jf:
		    jf.write(job)
		y=yaml.safe_load(job)
		job=batch_api.create_namespaced_job(namespace,y)

	def workers_status(self):
		os.system('kubectl get pods | grep ^tuner-')

