#!/usr/bin/env python3

import os
import sys
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream
from pathlib import Path

import uuid
import time

image = 'cerit.io/ljocha/gromacs:2023-2-plumed-2-9-afed-pytorch-model-cv-2'

try:
    with open(os.environ['HOME'] + '/.gmxjob') as j:
        jobname = j.read().strip()

except FileNotFoundError:
    jobname = 'asmsa-gmx-'+str(uuid.uuid4())
    with open(os.environ['HOME'] + '/.gmxjob','w') as j:
        j.write(jobname)

config.load_incluster_config()
v1 = client.CoreV1Api()

with open('/run/secrets/kubernetes.io/serviceaccount/namespace') as n:
    ns=n.read()

pod = v1.read_namespaced_pod(name=os.environ['HOSTNAME'], namespace=ns)

vol=[v.name for v in pod.spec.containers[0].volume_mounts if v.mount_path == os.environ['HOME']][0]

pvc = [ v for v in pod.spec.volumes if v.name == vol ][0].persistent_volume_claim.claim_name

batch_v1 = client.BatchV1Api()

def get_deepest_mountpoint(path: str) -> str:
    abs_path = os.path.abspath(path)
    current_stat = os.stat(abs_path)  
    current_dev = current_stat.st_dev
    current_path = abs_path
    while True:
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path: break
        parent_stat = os.stat(parent_path)
        parent_dev = parent_stat.st_dev

        if parent_dev != current_dev: return current_path
        current_path = parent_path
        current_dev = parent_dev
    return "/"

def create_job_object(job_name, image, pvc_name, namespace):
    container_security_context = client.V1SecurityContext(
        run_as_user=1000,
        run_as_group=1000,
        allow_privilege_escalation=False,
        capabilities=client.V1Capabilities(
            drop=['ALL']
        )
    )
    resources = client.V1ResourceRequirements(
        requests={
            "cpu": ".1",
            "memory": "8Gi",
        },
        limits={
            "cpu": "1",
            "memory": "8Gi",
        }
    )
    container = client.V1Container(
        name=job_name,
        image=image,
        working_dir="/mnt/",
        command=['sleep', 'inf'],
        security_context=container_security_context,
        resources=resources,
        env=[
            client.V1EnvVar(name='OMP_NUM_THREADS', value="1")
        ],
        volume_mounts=[
            client.V1VolumeMount(name='vol-1', mount_path='/mnt')
        ]
    )
    volume = client.V1Volume(
        name='vol-1',
        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
            claim_name=pvc_name
        )
    )
    pod_security_context = client.V1PodSecurityContext(
        fs_group_change_policy='OnRootMismatch',
        run_as_non_root=True,
        seccomp_profile=client.V1SeccompProfile(type='RuntimeDefault')
    )
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"job": job_name}),
        spec=client.V1PodSpec(
            restart_policy='Never',
            security_context=pod_security_context,
            containers=[container],
            volumes=[volume]
        )
    )
    job_spec = client.V1JobSpec(
        backoff_limit=0,
        template=template
    )
    metadata = client.V1ObjectMeta(name=job_name, namespace=namespace)
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=metadata,
        spec=job_spec
    )
    return job

mnt = get_deepest_mountpoint(os.getcwd())
base_path = Path(mnt).resolve()
target_path = Path(os.getcwd()).resolve()
relative_path = str(target_path.relative_to(base_path))

cmd = ['bash', '-c', f'cd {relative_path} && gmx '+" ".join([f"'{a}'" for a in sys.argv[1:]])]

try:
    batch_v1.read_namespaced_job(name=jobname, namespace=ns)
except ApiException as e:
    if e.status == 404:
        job_object = create_job_object(jobname, image, pvc, ns)
        api_response = batch_v1.create_namespaced_job(
            body=job_object,
            namespace=ns
        )
    else:
        raise e

while True:
    pods = v1.list_namespaced_pod(ns, label_selector=f"job={jobname}")
    if len(pods.items) > 0 and pods.items[0].status.phase == 'Running': break
    print(pods.items[0].status.phase)
    time.sleep(2)

resp = stream(
    v1.connect_get_namespaced_pod_exec,
    name=pods.items[0].metadata.name,
    namespace=ns,
    container=jobname,
    command=cmd,
    stderr=True, stdin=False,
    stdout=True, tty=False,
    _preload_content=True
)

print(resp)
