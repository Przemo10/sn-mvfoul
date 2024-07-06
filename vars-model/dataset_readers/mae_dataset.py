from torch.utils.data import Dataset
from random import random
import torch
import random
from data_loader import label2vectormerge, clips2vectormerge
from torchvision.io.video import read_video
from data_loader import get_ordered_random_indices
from transformers import AutoImageProcessor
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import numpy as np
import cv2


class MultiViewMAEDataset(Dataset):
    def __init__(self, path, start, end, fps, split, num_views, transform=None):

        if split != 'chall':
            # To load the annotations
            self.labels_offence_severity, self.labels_action, self.distribution_offence_severity, self.distribution_action, not_taking, self.number_of_actions = label2vectormerge(
                path, split, num_views)
            self.clips = clips2vectormerge(path, split, num_views, not_taking)
            self.clips = self.clips
            self.distribution_offence_severity = torch.div(self.distribution_offence_severity,
                                                           len(self.labels_offence_severity))
            self.distribution_action = torch.div(self.distribution_action, len(self.labels_action))

            self.weights_offence_severity = torch.div(1, self.distribution_offence_severity)
            self.weights_action = torch.div(1, self.distribution_action)
        else:
            self.clips = clips2vectormerge(path, split, num_views, [])

        # INFORMATION ABOUT SELF.LABELS_OFFENCE_SEVERITY
        # self.labels_offence_severity => Tensor of size of the dataset.
        # each element of self.labels_offence_severity is another tensor of size 4 (the number of classes) where the value is 1 if it is the class and 0 otherwise
        # for example if it is not an offence, then the tensor is [1, 0, 0, 0].

        # INFORMATION ABOUT SELF.LABELS_ACTION
        # self.labels_action => Tensor of size of the dataset.
        # each element of self.labels_action is another tensor of size 8 (the number of classes) where the value is 1 if it is the class and 0 otherwise
        # for example if the action is a tackling, then the tensor is [1, 0, 0, 0, 0, 0, 0, 0].

        # INFORMATION ABOUT SLEF.CLIPS
        # self.clips => list of the size of the dataset
        # each element of the list is another list of size of the number of views. The list contains the paths to all the views of that particular action.

        # The offence_severity groundtruth of the i-th action in self.clips, is the i-th element in the self.labels_offence_severity tensor
        # The type of action groundtruth of the i-th action in self.clips, is the i-th element in the self.labels_action tensor

        self.split = split
        self.start = start
        self.end = end
        self.transform = transform
        self.crop_margin = True

        self.transform_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
        if transform is not None:
            self.transform_model.mean = [0.485, 0.456, 0.406]
            self.transform_model.std = [ 0.229, 0.224, 0.225]
        self.num_views = num_views
        self.model_frames = 16
        self.fps = fps

        self.factor = (end - start) / (((end - start) / 25) * self.model_frames)

        self.length = len(self.clips)
        print(self.length)

    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action,

    def getWeights(self):
        return self.weights_offence_severity, self.weights_action,

        # RETURNS

    #
    # self.labels_offence_severity[index][0] => tensor of size 4. Example [1, 0, 0, 0] if the action is not an offence
    # self.labels_action[index][0] => tensor of size 8.           Example [1, 0, 0, 0, 0, 0, 0, 0] if the type of action is a tackling
    # videos => tensor of shape V, C, N, H, W with V = number of views, C = number of channels, N = the number of frames, H & W = height & width
    # self.number_of_actions[index] => the id of the action
    #
    def __getitem__(self, index):

        prev_views = []

        for num_view in range(len(self.clips[index])):

            index_view = num_view

            if len(prev_views) == 2:
                continue

            # As we use a batch size > 1 during training, we always randomly select two views even if we have more than two views.
            # As the batch size during validation and testing is 1, we can have 2, 3 or 4 views per action.
            cont = True
            if self.split == 'train':
                while cont:
                    aux = random.randint(0, len(self.clips[index]) - 1)
                    if aux not in prev_views:
                        cont = False
                index_view = aux
                prev_views.append(index_view)

            cap = cv2.VideoCapture(self.clips[index][index_view])
            frames = []

            # Read until video is completed
            while cap.isOpened():
                # Capture frame-by-frame
                ret, frame = cap.read()
                if not ret:
                    break
                # Append each frame to the list
                frames.append(frame)

            # Convert the list of frames into a numpy array
            video_np = np.array(frames, dtype=np.uint8)

            # Release the VideoCapture object
            cap.release()


            if self.fps != 16:
                indices = get_ordered_random_indices(self.fps)
                final_frames = video_np[indices, :, :, :]
            else:
                final_frames = video_np[self.start:self.end, :, :, :]
            if self.crop_margin:
                final_frames = final_frames[:,25:, 25:, :]

            final_frames = torch.tensor(final_frames).permute(0,3,1,2)

            if self.transform:
                final_frames = self.transform(final_frames)

            final_frames = self.transform_model(final_frames)
            final_frames = final_frames.permute(1,0,2,3)

            if num_view == 0:
                videos = final_frames.unsqueeze(0)
            else:

                videos = torch.cat((videos, final_frames.unsqueeze(0)), 0)

        if self.num_views != 1 and self.num_views != 5:
            videos = videos.squeeze()

        if self.split != 'chall':
            return self.labels_offence_severity[index][0], self.labels_action[index][0], videos, self.number_of_actions[
                index]
        else:
            return -1, -1, videos, str(index)

    def __len__(self):
        return self.length

