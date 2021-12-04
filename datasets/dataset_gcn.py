import json
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SkeletonDataset(Dataset):
    def __init__(self, annotation_dict, npy_dir="../datasets/npy/", video_dir="../datasets/mp4/", transform=None):
        with open(annotation_dict) as f:
            self.video_list = list(json.load(f).items())

        self.num_node = 18

        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [
            (1, 0), (2, 1), (3, 2), (4, 3), (5, 1),
            (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
            (11, 5), (12, 11), (13, 12), (14, 0), (15, 0),
            (16, 14), (17, 15)
        ]
        self.bone_link = [
            (0, 0),
            (1, 0), (2, 1), (3, 2), (4, 3), (5, 1),
            (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
            (11, 5), (12, 11), (13, 12), (14, 0), (15, 0),
            (16, 14), (17, 15)
        ]
        self.edge = neighbor_link + self_link
        self.center = 1
        self.adjacency = self.get_adjacency_matrix(self.num_node, self.edge)

        self.npy_dir = npy_dir
        self.video_dir = video_dir
        self.transform = transform

    def __len__(self):
        # return length of none-flipped videos in directory
        return len(self.video_list)

    def __getitem__(self, idx):
        video_id = self.video_list[idx][0]
        class_num = self.video_list[idx][1]
        encoding = np.squeeze(
            np.eye(10)[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1)])

        # use joint data
        joints = np.load(self.npy_dir + video_id +
                         ".npy", allow_pickle=True)
        # use video data
        video = self.VideoToNumpy(video_id)

        # node prerocessing
        for index in range(len(joints)):
            for i, (start_p, end_p) in enumerate(self.edge):
                if joints[index].get(start_p) == None and joints[index].get(end_p) == None:
                    joints[index][start_p] = 1
                    joints[index][end_p] = 1
                if joints[index].get(start_p) == None:
                    joints[index][start_p] = joints[index][end_p]
                if joints[index].get(end_p) == None:
                    joints[index][end_p] = joints[index][start_p]
            joints[index] = dict(sorted(joints[index].items()))

        # Adjancey matrix
        adjacency = self.adjacency

        # joint feature
        joint_feature = np.empty([16, 18, 2])
        for i in range(0, 16):
            for j in range(0, 18):
                joint_feature[i][j] = joints[i][j]

        # bone feature
        bone_feature = np.empty([16, 18, 2])
        for i in range(0, 16):
            for j, (v1, v2) in enumerate(self.bone_link):
                bone_feature[i][j] = tuple(
                    map(lambda i, j: i - j, joints[i][v1], joints[i][v2]))

        # joint bone feature
        joint_bone_feature = np.empty([16, 18, 4])
        for i in range(0, 16):
            for j in range(0, 18):
                joint_bone_feature[i][j] = np.append(
                    joint_feature[i][j], bone_feature[i][j])

        # normalize
        joint_bone_feature[:, :, 0] = 2 * \
            ((joint_bone_feature[:, :, 0] - 0) / (128 - 0)) - 1
        joint_bone_feature[:, :, 1] = 2 * \
            ((joint_bone_feature[:, :, 1] - 0) / (176 - 0)) - 1

        sample = {
            'video_id': video_id,
            'video': torch.from_numpy(video).float(),
            'node_features': torch.from_numpy(joint_feature).float(),
            'adjacency': adjacency,
            'action': torch.from_numpy(np.array(encoding[class_num])),
            'class': class_num,
        }

        return sample

    def VideoToNumpy(self, video_id):
        # get video
        video = cv2.VideoCapture(self.video_dir + video_id + ".mp4")

        if not video.isOpened():
            video = cv2.VideoCapture(self.augmented_dir + video_id + ".mp4")
        if not video.isOpened():
            raise Exception("Video file not readable")

        video_frames = []
        while (video.isOpened()):
            # read video
            success, frame = video.read()
            if not success:
                break

            frame = np.asarray([frame[..., i]
                               for i in range(frame.shape[-1])]).astype(float)
            video_frames.append(frame)

        video.release()
        assert len(video_frames) == 16
        # (N, C, H, W) -> (C, N, H, W)
        return np.transpose(np.asarray(video_frames), (1, 0, 2, 3))

    def get_adjacency_matrix(self, num_node, edge):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        return A


if __name__ == '__main__':
    skeleton_dataset = SkeletonDataset(
        annotation_dict="../datasets/annotation_dict.json")
