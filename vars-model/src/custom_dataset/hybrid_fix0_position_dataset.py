from torch.utils.data import Dataset
from random import random
import torch
import random
from src.custom_dataset.data_loader import label2vectormerge, clips2vectormerge, create_inverse_proportion_exp_fun_weights
from torchvision.io.video import read_video


class MultiViewDatasetHybrid(Dataset):
    def __init__(self, path, start, end, fps, split, num_views, transform=None, transform_model=None,video_shift_aug=0,
                 weight_exp_alpha = 8.0, weight_exp_bias = 0.02, weight_exp_gamma =2.0, crop25 =0):

        if split != 'chall':
            # To load the annotations
            self.labels_offence_severity, self.labels_action, self.distribution_offence_severity, self.distribution_action, not_taking, self.number_of_actions = label2vectormerge(
                path, split, num_views)
            self.clips = clips2vectormerge(path, split, num_views, not_taking)
            self.distribution_offence_severity = torch.div(self.distribution_offence_severity,
                                                           len(self.labels_offence_severity))
            self.distribution_action = torch.div(self.distribution_action, len(self.labels_action))

            self.weights_offence_severity = torch.div(1, self.distribution_offence_severity)
            self.weights_action = torch.div(1, self.distribution_action)
            self.weights_inverse_exp_offence_severity = create_inverse_proportion_exp_fun_weights(
                self.distribution_offence_severity * len(self.labels_offence_severity),
                alpha=weight_exp_alpha,
                bias_value=weight_exp_bias,
                gamma=weight_exp_gamma
            )
            self.weights_inverse_exp_action = create_inverse_proportion_exp_fun_weights(
                self.distribution_action * len(self.labels_action),
                alpha=weight_exp_alpha,
                bias_value=weight_exp_bias,
                gamma=weight_exp_gamma
            )

            self.video_shift_aug = video_shift_aug
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
        self.transform_model = transform_model
        self.num_views = num_views
        self.factor = (end - start) / (((end - start) / 25) * fps)
        self.crop25 =crop25

        self.length = len(self.clips)
        print(self.length)

    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action,

    def getWeights(self):
        return self.weights_offence_severity, self.weights_action,

        # RETURNS
    def getExpotentialWeight(self):
        return self.weights_inverse_exp_offence_severity, self.weights_inverse_exp_action
    #
    # self.labels_offence_severity[index][0] => tensor of size 4. Example [1, 0, 0, 0] if the action is not an offence
    # self.labels_action[index][0] => tensor of size 8.           Example [1, 0, 0, 0, 0, 0, 0, 0] if the type of action is a tackling
    # videos => tensor of shape V, C, N, H, W with V = number of views, C = number of channels, N = the number of frames, H & W = height & width
    # self.number_of_actions[index] => the id of the action
    #
    def __getitem__(self, index):


        total_clips = len(self.clips[index])

        if self.split != 'c1' and self.num_views == 2:
            prev_views = [0]
            v = random.randint(1, len(self.clips[index])-1)
            prev_views.append(v)
        elif self.split != 'c2' and self.num_views == 3:
            if total_clips == 2:
                prev_views = [0, 1, 1]
            else:
                prev_views = [0, *random.sample(range(1, total_clips),2)]

        elif self.split != 'c2' and self.num_views == 4:
            if total_clips == 2:
                prev_views = [0, 1, 1, 1]
            elif total_clips == 3:
                prev_views = [0, *random.choices([1,2],k=3)]

            else:
                prev_views = [0, *random.sample(range(1,total_clips),3)]
        else:
            prev_views = [range(total_clips)]

        for num_view in prev_views:

            video, _, _ = read_video(self.clips[index][num_view], output_format="THWC", pts_unit='sec')

            if self.split == 'train' and self.video_shift_aug > 0:
                rand_shift = random.randint(-self.video_shift_aug, 2)
                start = self.start + rand_shift
                end = self.end + rand_shift
                frames = video[start:end, :, :, :]
            else:
                frames = video[self.start:self.end, :, :, :]

            final_frames = None

            for j in range(len(frames)):
                if j % self.factor < 1:
                    if final_frames == None:
                        final_frames = frames[j, :, :, :].unsqueeze(0)
                    else:
                        final_frames = torch.cat((final_frames, frames[j, :, :, :].unsqueeze(0)), 0)

            final_frames = final_frames.permute(0, 3, 1, 2)

            if self.transform != None:
                final_frames = self.transform(final_frames)

            final_frames = self.transform_model(final_frames)
            final_frames = final_frames.permute(1, 0, 2, 3)

            if num_view == 0:
                videos = final_frames.unsqueeze(0)
            else:
                final_frames = final_frames.unsqueeze(0)
                videos = torch.cat((videos, final_frames), 0)

        if self.num_views != 1 and self.num_views != 5:
            videos = videos.squeeze()

        videos = videos.permute(0, 2, 1, 3, 4)

        if self.split != 'chall':
            return self.labels_offence_severity[index][0], self.labels_action[index][0], videos, self.number_of_actions[
                index]
        else:
            return -1, -1, videos, str(index)

    def __len__(self):
        return self.length