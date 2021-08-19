#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import jittor as jt
import jittor.nn as nn

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from subdivnet.network import MeshNet
from subdivnet.utils import to_mesh_tensor

class SubDivNetDetector:
    def __init__(self):
        self.reset()

        jt.flags.use_cuda = 1
        return

    def reset(self):
        self.model = None
        self.model_ready = False
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0

        self.input_channels = None
        self.n_classes = None
        self.depth = 3
        self.channels = [32, 64, 128, 256]
        self.n_dropout = 2
        self.use_xyz = True
        self.use_normal = True
        self.checkpoint = None

        self.residual = False
        self.blocks = None
        self.no_center_diff = False
        self.feats = ['area', 'face_angles', 'curvs', 'center', 'normal']
        return

    def resetTimer(self):
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0
        return

    def startTimer(self):
        self.time_start = time.time()
        return

    def endTimer(self, save_time=True):
        time_end = time.time()

        if not save_time:
            return

        if self.time_start is None:
            print("startTimer must run first!")
            return

        if time_end > self.time_start:
            self.total_time_sum += time_end - self.time_start
            self.detected_num += 1
        else:
            print("Time end must > time start!")
        return

    def getAverageTimeMS(self):
        if self.detected_num == 0:
            return -1

        return int(1000.0 * self.total_time_sum / self.detected_num)

    def getAverageFPS(self):
        if self.detected_num == 0:
            return -1

        return int(1.0 * self.detected_num / self.total_time_sum)

    def randomizeMeshOrientation(self, mesh: trimesh.Trimesh):
        axis_seq = ''.join(random.sample('xyz', 3))
        angles = [random.choice([0, 90, 180, 270]) for _ in range(3)]
        rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
        mesh.vertices = rotation.apply(mesh.vertices)
        return mesh

    def randomScale(self, mesh: trimesh.Trimesh):
        mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
        return mesh

    def meshNormalize(self, mesh: trimesh.Trimesh):
        vertices = mesh.vertices - mesh.vertices.min(axis=0)
        vertices = vertices / vertices.max()
        mesh.vertices = vertices
        return mesh

    def loadMesh(self, path, normalize=False, augments=[], request=[]):
        mesh = trimesh.load_mesh(path, process=False)

        for method in augments:
            if method == 'orient':
                mesh = self.randomizeMeshOrientation(mesh)
            if method == 'scale':
                mesh = self.randomScale(mesh)

        if normalize:
            mesh = self.meshNormalize(mesh)

        F = mesh.faces
        V = mesh.vertices
        Fs = mesh.faces.shape[0]

        face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
        # corner = V[F.flatten()].reshape(-1, 3, 3) - face_center[:, np.newaxis, :]
        vertex_normals = mesh.vertex_normals
        face_normals = mesh.face_normals
        face_curvs = np.vstack([
            (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
            (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
            (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
        ])
        
        feats = []
        if 'area' in request:
            feats.append(mesh.area_faces)
        if 'normal' in request:
            feats.append(face_normals.T)
        if 'center' in request:
            feats.append(face_center.T)
        if 'face_angles' in request:
            feats.append(np.sort(mesh.face_angles, axis=1).T)
        if 'curvs' in request:
            feats.append(np.sort(face_curvs, axis=0))

        feats = np.vstack(feats)
        return mesh.faces, feats, Fs

    def createMeshes(self, mesh_file_path):
        faces, feats, Fs = self.loadMesh(mesh_file_path,
                                         normalize=True,
                                         augments=[],
                                         request=self.feats)

        np_faces = np.zeros((1, Fs, 3), dtype=np.int32)
        np_feats = np.zeros((1, feats.shape[0], Fs), dtype=np.float32)
        np_Fs = np.int32([Fs])

        np_faces[0, :Fs] = faces
        np_feats[0, :, :Fs] = feats

        meshes = {
            'faces': np_faces,
            'feats': np_feats,
            'Fs': np_Fs
        }
        return meshes

    def setCheckPoint(self, checkpoint):
        self.checkpoint = checkpoint
        return

    def initEnv(self, checkpoint, n_classes):
        self.reset()
        self.setCheckPoint(checkpoint)
        self.n_classes = n_classes

        self.input_channels = 7
        if self.use_xyz:
            self.input_channels += 3
        if self.use_normal:
            self.input_channels += 3
        return

    def loadModel(self, checkpoint, n_classes):
        self.initEnv(checkpoint, n_classes)

        self.model = MeshNet(
            self.input_channels,
            out_channels=self.n_classes, depth=self.depth, 
            layer_channels=self.channels, residual=self.residual, 
            blocks=self.blocks, n_dropout=self.n_dropout, 
            center_diff=not self.no_center_diff)

        if self.checkpoint is not None:
            self.model.load(self.checkpoint)

        self.model.eval()

        self.model_ready = True
        return

    def detect(self, mesh):
        with jt.no_grad():
            mesh_tensor = to_mesh_tensor(mesh)
            result = self.model(mesh_tensor)
            return result


    @jt.single_process_scope()
    def test(self, mesh_folder_path, run_episode=-1, timer_skip_num=5):
        if not self.model_ready:
            print("Model not ready yet, Please loadModel or check your model path first!")
            return

        if run_episode == 0:
            print("No detect run with run_episode=0!")
            return

        file_name_list = os.listdir(mesh_folder_path)
        mesh_file_name_list = []
        for file_name in file_name_list:
            if file_name[-4:] == ".obj":
                mesh_file_name_list.append(file_name)

        if run_episode < 0:
            self.resetTimer()
            timer_skipped_num = 0

            while True:
                for mesh_file_name in mesh_file_name_list:
                    mesh_file_path = os.path.join(mesh_folder_path, mesh_file_name)

                    self.startTimer()

                    meshes = self.createMeshes(mesh_file_path)

                    result = self.detect(meshes)

                    preds = np.argmax(result.data, axis=1)

                    if timer_skipped_num < timer_skip_num:
                        self.endTimer(False)
                        timer_skipped_num += 1
                    else:
                        self.endTimer()

                    print("\rNet: SubDivNet" +
                          "\tDetected: " + str(self.detected_num) +
                          "\tAvgTime: " + str(self.getAverageTimeMS()) + "ms"
                          "\tAvgFPS: " + str(self.getAverageFPS()) +
                          "    ", end="")

            print()

            return

        self.resetTimer()
        total_num = run_episode * len(mesh_file_name_list)
        timer_skipped_num = 0

        for i in range(run_episode):
            for mesh_file_name in mesh_file_name_list:
                mesh_file_path = os.path.join(mesh_folder_path, mesh_file_name)

                self.startTimer()

                meshes = self.createMeshes(mesh_file_path)

                result = self.detect(meshes)

                preds = np.argmax(result.data, axis=1)

                if timer_skipped_num < timer_skip_num:
                    self.endTimer(False)
                    timer_skipped_num += 1
                else:
                    self.endTimer()

                print("\rNet: SubDivNet" +
                      "\tDetected: " + str(self.detected_num) + "/" + str(total_num - timer_skip_num) +
                      "\t\tAvgTime: " + str(self.getAverageTimeMS()) + "ms"
                      "\tAvgFPS: " + str(self.getAverageFPS()) +
                      "    ", end="")

        print()


        print("pred  : ", preds)

if __name__ == '__main__':
    checkpoint = "/home/chli/github/SubdivNet/checkpoints/Manifold40.pkl"
    n_classes = 40
    mesh_folder_path = "/home/chli/github/SubdivNet/data/Manifold40-MAPS-96-3/bed/test/"

    subdivnet_detector = SubDivNetDetector()

    subdivnet_detector.loadModel(checkpoint, n_classes)

    subdivnet_detector.test(mesh_folder_path, -1)

