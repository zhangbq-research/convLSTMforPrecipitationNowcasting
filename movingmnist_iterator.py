import logging
import cv2
import numpy as np
import math
import os
from sklearn import preprocessing

logger = logging.getLogger(__name__)


def load_mnist(training_num=50000):
    """Load the mnist dataset

    Parameters
    ----------
    training_num

    Returns
    -------

    """
    import sys
    
    MNIST_PATH = 'E:/data/mnist_data/'
    data_path = os.path.join(MNIST_PATH, "mnist.npz")
    if not os.path.isfile(data_path):
        origin = (
            'https://github.com/sxjscience/mxnet/raw/master/example/bayesian-methods/mnist.npz'
        )
        print('Downloading data from %s to %s' % (origin, data_path))

        from urllib import request
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context  # Not verify
        data_file = request.urlopen(origin)
        with open(data_path, 'wb') as output:
            output.write(data_file.read())
        print('Done!')
    dat = np.load(data_path)
    X = dat['X'][:training_num]
    Y = dat['Y'][:training_num]
    X_test = dat['X_test']
    Y_test = dat['Y_test']
    Y = Y.reshape((Y.shape[0],))
    Y_test = Y_test.reshape((Y_test.shape[0],))
    return X, Y, X_test, Y_test


def move_step(v0, p0, bounding_box):
    xmin, xmax, ymin, ymax = bounding_box
    assert (p0[0] >= xmin) and (p0[0] <= xmax) and (p0[1] >= ymin) and (p0[1] <= ymax)
    v = v0.copy()
    assert v[0] != 0.0 and v[1] != 0.0
    p = v0 + p0
    while (p[0] < xmin) or (p[0] > xmax) or (p[1] < ymin) or (p[1] > ymax):
        vx, vy = v
        x, y = p
        dist = np.zeros((4,))
        dist[0] = abs(x - xmin) if ymin <= (xmin - x) * vy / vx + y <= ymax else np.inf
        dist[1] = abs(x - xmax) if ymin <= (xmax - x) * vy / vx + y <= ymax else np.inf
        dist[2] = abs((y - ymin) * vx / vy) if xmin <= (ymin - y) * vx / vy + x <= xmax else np.inf
        dist[3] = abs((y - ymax) * vx / vy) if xmin <= (ymax - y) * vx / vy + x <= xmax else np.inf
        n = np.argmin(dist)
        if n == 0:
            v[0] = -v[0]
            p[0] = 2 * xmin - p[0]
        elif n == 1:
            v[0] = -v[0]
            p[0] = 2 * xmax - p[0]
        elif n == 2:
            v[1] = -v[1]
            p[1] = 2 * ymin - p[1]
        elif n == 3:
            v[1] = -v[1]
            p[1] = 2 * ymax - p[1]
        else:
            assert False
    return v, p


def crop_mnist_digit(digit_img, tol=5):
    """Return the cropped version of the mnist digit

    Parameters
    ----------
    digit_img : np.ndarray
        Shape: ()

    Returns
    -------

    """
    tol = float(tol) / float(255)
    mask = digit_img > tol
    return digit_img[np.ix_(mask.any(1), mask.any(0))]

class MovingMNISTAdvancedIterator(object):

    def __init__(self,is_CGAN=False,
                 digit_num=3,
                 distractor_num=0,
                 img_size=64,
                 distractor_size=5,
                 max_velocity_scale=3.6,
                 initial_velocity_range=(0.0, 3.6),
                 acceleration_range=(0.0, 0.0),
                 scale_variation_range=(1 / 1.1, 1.1),
                 rotation_angle_range=(-30, 30),
                 global_rotation_angle_range=(-30, 30),
                 illumination_factor_range=(0.6, 1.0),
                 period=5,
                 global_rotation_prob=0.5,
                 index_range=(0, 40000)):
        """

        Parameters
        ----------
        digit_num : int
            Number of digits
        distractor_num : int
            Number of distractors
        img_size : int
            Size of the image
        distractor_size : int
            Size of the distractors
        max_velocity_scale : float
            Maximum scale of the velocity
        initial_velocity_range : tuple
        acceleration_range
        scale_variation_range
        rotation_angle_range
        period : period of the
        index_range
        """
        self.is_CGAN = is_CGAN
        self.mnist_train_img, self.mnist_train_label,\
        self.mnist_test_img, self.mnist_test_label = load_mnist()
        self._digit_num = digit_num
        self._img_size = img_size
        self._distractor_size = distractor_size
        self._distractor_num = distractor_num
        self._max_velocity_scale = max_velocity_scale
        self._initial_velocity_range = initial_velocity_range
        self._acceleration_range = acceleration_range
        self._scale_variation_range = scale_variation_range
        self._rotation_angle_range = rotation_angle_range
        self._illumination_factor_range = illumination_factor_range
        self._period = period
        self._global_rotation_angle_range = global_rotation_angle_range
        self._global_rotation_prob = global_rotation_prob
        self._index_range = index_range
        self._h5py_f = None
        self._seq = None
        self._motion_vectors = None
        self.replay = None
        self.replay_index = 0
        self.replay_numsamples = -1

    def _choose_distractors(self, distractor_seeds):
        """Choose the distractors

        We use the similar approach as
         https://github.com/deepmind/mnist-cluttered/blob/master/mnist_cluttered.lua
        Returns
        -------
        ret : list
            list of distractor images
        """
        ret = []
        for i in range(self._distractor_num):
            ind = math.floor(distractor_seeds[i, 2] * self._index_range[1])
            distractor_img = self.mnist_train_img[ind].reshape((28, 28))
            distractor_h_begin = math.floor(distractor_seeds[i, 3] * (28 - self._distractor_size))
            distractor_w_begin = math.floor(distractor_seeds[i, 4] * (28 - self._distractor_size))
            distractor_img = distractor_img[
                distractor_h_begin:distractor_h_begin + self._distractor_size,
                distractor_w_begin:distractor_w_begin + self._distractor_size]
            ret.append(distractor_img)
        return ret

    def draw_distractors(self, canvas_img, distractor_seeds):
        """

        Parameters
        ----------
        canvas_img

        Returns
        -------

        """

        distractor_imgs = self._choose_distractors(distractor_seeds)

        for i, img in enumerate(distractor_imgs):
            r_begin = math.floor(distractor_seeds[i][0] * (self._img_size - img.shape[0]))
            c_begin = math.floor(distractor_seeds[i][1] * (self._img_size - img.shape[1]))
            canvas_img[r_begin:r_begin + img.shape[0], c_begin:c_begin +
                       img.shape[1]] = img
        return canvas_img

    def draw_imgs(self,
                  base_img,
                  affine_transforms,
                  prev_affine_transforms=None):
        """

        Parameters
        ----------
        base_img : list
            Inner Shape: (H, W)
        affine_transforms : np.ndarray
            Shape: (digit_num, 2, 3)
        prev_affine_transforms : np.ndarray
            Shape: (digit_num, 2, 3)

        Returns
        -------

        """
        canvas_img = np.zeros(
            (self._img_size, self._img_size), dtype=np.float32)
        for i in range(self._digit_num):
            tmp_img = cv2.warpAffine(base_img[i], affine_transforms[i],
                                     (self._img_size, self._img_size))
            canvas_img = np.maximum(canvas_img, tmp_img)
        return canvas_img

    def _find_center(self, img):
        x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
        raise NotImplementedError

    def _bounce_border(self, inner_boundary, affine_transform, digit_shift,
                       velocity, img_h, img_w):
        # top-left, top-right, down-left, down-right
        center = affine_transform.dot(
            np.array([img_w / 2.0, img_h / 2.0, 1], dtype=np.float32))
        new_velocity = velocity.copy()
        new_center = center.copy()
        if center[0] < inner_boundary[0]:
            new_velocity[0] = -new_velocity[0]
            new_center[0] = inner_boundary[0]
        if center[0] > inner_boundary[2]:
            new_velocity[0] = -new_velocity[0]
            new_center[0] = inner_boundary[2]
        if center[1] < inner_boundary[1]:
            new_velocity[1] = -new_velocity[1]
            new_center[1] = inner_boundary[1]
        if center[1] > inner_boundary[3]:
            new_velocity[1] = -new_velocity[1]
            new_center[1] = inner_boundary[3]
        affine_transform[:, 2] += new_center - center
        digit_shift += new_center - center
        return affine_transform, digit_shift, new_velocity

    def get_label_images(self,digitals,width=64,height=64):
        labels = []
        one_hot = np.zeros((10))
        for i in range(10):

            if i in digitals:
                one_hot[i]=1
                labels.append(np.ones((width,height)))
            else:
                labels.append(np.zeros((width,height)))
        labels = np.array(labels).transpose((1,2,0))
        return labels,one_hot

    def sample(self, batch_size, in_len,out_len, random=True):
        """

        Parameters
        ----------
        batch_size : int
        seqlen : int
        random: take random samples from loaded parameters. Ignored if no parameters are loaded.

        Returns
        -------
        seq : np.ndarray
            Shape: (seqlen, batch_size, 1, H, W)
        motion_vectors : np.ndarray
            Shape: (seqlen, batch_size, 2, H, W)
        """
        seqlen = in_len+out_len
        if self.replay is not None:
            if random is True:
                self.replay_index = np.random.randint(self.replay_numsamples - batch_size)
            elif self.replay_index + batch_size > self.replay_numsamples:
                raise IndexError("Not enough pre-generated parameters to create new sample.")

        seq = np.zeros(
            (seqlen, batch_size, self._img_size, self._img_size),
            dtype=np.float32)
        motion_vectors = np.zeros(
            (seqlen, batch_size, 2, self._img_size, self._img_size),
            dtype=np.float32)
        inner_boundary = np.array(
            [10, 10, self._img_size - 10, self._img_size - 10],
            dtype=np.float32)
        la = []
        lables_one_hot = []
        for b in range(batch_size):
            affine_transforms = np.zeros(
                (seqlen, self._digit_num, 2, 3), dtype=np.float32)
            appearance_variants = np.ones(
                (seqlen, self._digit_num), dtype=np.float32)
            scale = np.ones((seqlen, self._digit_num), dtype=np.float32)
            rotation_angle = np.zeros(
                (seqlen, self._digit_num), dtype=np.float32)
            init_velocity = np.zeros(
                shape=(self._digit_num, 2), dtype=np.float32)
            velocity = np.zeros((seqlen, self._digit_num, 2), dtype=np.float32)
            digit_shift = np.zeros(
                (seqlen, self._digit_num, 2), dtype=np.float32)

            if self.replay is not None:
                digit_indices = self.replay["digit_indices"][self.replay_index
                                                             + b]
                appearance_mult = self.replay["appearance_mult"][
                    self.replay_index + b]
                scale_variation = self.replay["scale_variation"][
                    self.replay_index + b]
                base_rotation_angle = self.replay["base_rotation_angle"][
                    self.replay_index + b]
                affine_transforms_multipliers = self.replay[
                    "affine_transforms_multipliers"][self.replay_index + b]
                init_velocity_angle = self.replay["init_velocity_angle"][
                    self.replay_index + b]
                init_velocity_magnitude = self.replay[
                    "init_velocity_magnitude"][self.replay_index + b]
                distractor_seeds = self.replay[
                    "distractor_seeds"][self.replay_index + b]

                assert(distractor_seeds.shape[0] == seqlen)

            else:
                digit_indices = np.random.randint(
                    low=self._index_range[0],
                    high=self._index_range[1],
                    size=self._digit_num)
                appearance_mult = np.random.uniform(
                    low=self._illumination_factor_range[0],
                    high=self._illumination_factor_range[1])
                scale_variation = np.random.uniform(
                    low=self._scale_variation_range[0],
                    high=self._scale_variation_range[1],
                    size=(self._digit_num, ))
                base_rotation_angle = np.random.uniform(
                    low=self._rotation_angle_range[0],
                    high=self._rotation_angle_range[1],
                    size=(self._digit_num, ))
                affine_transforms_multipliers = np.random.uniform(
                    size=(self._digit_num, 2))
                init_velocity_angle = np.random.uniform(size=(
                    self._digit_num, )) * (2 * np.pi)
                init_velocity_magnitude = np.random.uniform(
                    low=self._initial_velocity_range[0],
                    high=self._initial_velocity_range[1],
                    size=self._digit_num)
                distractor_seeds = np.random.uniform(
                    size=(seqlen, self._distractor_num, 5))

            base_digit_img = [
                crop_mnist_digit(self.mnist_train_img[i].reshape((28, 28)))
                for i in digit_indices
            ]
            lables_digital = [
                self.mnist_train_label[i]
                for i in digit_indices
            ]
            l,one_hot = self.get_label_images(lables_digital)
            lables_one_hot.append(one_hot)
            la.append(l)

            for i in range(1, seqlen):
                appearance_variants[i, :] = appearance_variants[i - 1, :] *\
                                            (appearance_mult ** -(2 * ((i // 5) % 2) - 1))

            for i in range(1, seqlen):
                base_factor = (2 * ((i // 5) % 2) - 1)
                scale[i, :] = scale[i - 1, :] * (scale_variation**base_factor)
                rotation_angle[i, :] = rotation_angle[
                    i - 1, :] + base_rotation_angle

            affine_transforms[0, :, 0, 0] = 1.0
            affine_transforms[0, :, 1, 1] = 1.0
            for i in range(self._digit_num):
                affine_transforms[0, i, 0, 2] = affine_transforms_multipliers[i, 0] *\
                    (self._img_size - base_digit_img[i].shape[1])
                affine_transforms[0, i, 1, 2] = affine_transforms_multipliers[i, 1] *\
                    (self._img_size - base_digit_img[i].shape[0])

            init_velocity[:, 0] = init_velocity_magnitude * np.cos(
                init_velocity_angle)
            init_velocity[:, 1] = init_velocity_magnitude * np.sin(
                init_velocity_angle)
            curr_velocity = init_velocity

            # base_acceleration_angle = np.random.random() * 2 * np.pi
            # base_acceleration_magnitude = np.random.uniform(low=self._acceleration_range[0],
            #                                                 high=self._acceleration_range[1],
            #                                                 size=self._digit_num)
            # base_acceleration = np.zeros(shape=(self._digit_num, 2), dtype=np.float32)
            # base_acceleration[:, 0] = base_acceleration_magnitude * np.cos(init_velocity_angle)
            # base_acceleration[:, 1] = base_acceleration_magnitude * np.sin(init_velocity_angle)

            for i in range(self._digit_num):
                digit_shift[0, i, 0] = affine_transforms[
                    0, i, 0, 2]  #+ (base_digit_img[i].shape[1] / 2.0)
                digit_shift[0, i, 1] = affine_transforms[
                    0, i, 1, 2]  #+ (base_digit_img[i].shape[0] / 2.0)

            for i in range(seqlen - 1):
                velocity[i, :, :] = curr_velocity
                #curr_velocity += base_acceleration * (2 * ((i / 5) % 2) - 1)
                curr_velocity = np.clip(
                    curr_velocity,
                    a_min=-self._max_velocity_scale,
                    a_max=self._max_velocity_scale)
                for j in range(self._digit_num):
                    digit_shift[i + 1, j, :] = digit_shift[
                        i, j, :] + curr_velocity[j]
                    rotation_mat = cv2.getRotationMatrix2D(
                        center=(base_digit_img[j].shape[1] / 2.0,
                                base_digit_img[j].shape[0] / 2.0),
                        angle=rotation_angle[i + 1, j],
                        scale=scale[i + 1, j])
                    affine_transforms[i + 1, j, :, :2] = rotation_mat[:, :2]
                    affine_transforms[i + 1, j, :, 2] = digit_shift[
                        i + 1, j, :] + rotation_mat[:, 2]
                    affine_transforms[i + 1, j, :, :], digit_shift[i + 1, j, :], curr_velocity[j] =\
                        self._bounce_border(inner_boundary=inner_boundary,
                                            affine_transform=affine_transforms[i + 1, j, :, :],
                                            digit_shift=digit_shift[i + 1, j, :],
                                            velocity=curr_velocity[j],
                                            img_h=base_digit_img[j].shape[0],
                                            img_w=base_digit_img[j].shape[1])
            for i in range(seqlen):
                seq[i, b, :, :] = self.draw_imgs(
                    base_img=[
                        base_digit_img[j] * appearance_variants[i, j]
                        for j in range(self._digit_num)
                    ],
                    affine_transforms=affine_transforms[i])
                self.draw_distractors(seq[i, b, :, :], distractor_seeds[i])

        self.replay_index += batch_size

        # input_frames = seq[:in_len]
        # output_frames = seq[in_len:]
        #
        # input_frames = input_frames.transpose((1, 2, 3, 0))
        # output_frames = output_frames.transpose((1, 2, 3, 0))

        if self.is_CGAN:
            return seq,np.array(la),np.array(lables_one_hot)
        else:
            return seq

    def load(self, file):
        """Initialize to draw samples from pre-computed parameters.

        Args:
            file: Either the file name (string) or an open file (file-like
                object) from which the data will be loaded.
        """
        self.replay_index = 0
        with np.load(file) as f:
            self.replay = dict(f)

        assert(self.replay["distractor_seeds"].shape[2] == self._distractor_num)

        num_samples, seqlen = self.replay["distractor_seeds"].shape[0:2]
        self.replay_numsamples = num_samples
        return num_samples, seqlen

    def save(self, seqlen, num_samples=10000, file=None):
        """Draw random numbers for num_samples sequences and save them.

        This initializes the state of MovingMNISTAdvancedIterator to generate
        sequences based on the hereby drawn parameters.

        Note that each call to sample(batch_size, seqlen) will use batch_size
        of the num_samples parameters.

        Args:
            num_samples: Number of unique MovingMNISTAdvanced sequences to draw
                parameters for
            file: Either the file name (string) or an open file (file-like
                object) where the data will be saved. If file is a string or a
                Path, the .npz extension will be appended to the file name if
                it is not already there.

        """
        if file is None:
            file = "mnist_{}".format(num_samples)

        self.replay = dict()
        self.replay["digit_indices"] = np.random.randint(
            low=self._index_range[0],
            high=self._index_range[1],
            size=(num_samples, self._digit_num))
        self.replay["appearance_mult"] = np.random.uniform(
            low=self._illumination_factor_range[0],
            high=self._illumination_factor_range[1],
            size=(num_samples, ))
        self.replay["scale_variation"] = np.random.uniform(
            low=self._scale_variation_range[0],
            high=self._scale_variation_range[1],
            size=(num_samples, self._digit_num))
        self.replay["base_rotation_angle"] = np.random.uniform(
            low=self._rotation_angle_range[0],
            high=self._rotation_angle_range[1],
            size=(num_samples, self._digit_num))
        self.replay["affine_transforms_multipliers"] = np.random.uniform(
            size=(num_samples, self._digit_num, 2))
        self.replay["init_velocity_angle"] = np.random.uniform(
            size=(num_samples, self._digit_num)) * 2 * np.pi
        self.replay["init_velocity_magnitude"] = np.random.uniform(
            low=self._initial_velocity_range[0],
            high=self._initial_velocity_range[1],
            size=(num_samples, self._digit_num))
        self.replay["distractor_seeds"] = np.random.uniform(
            size=(num_samples, seqlen, self._distractor_num, 5))

        self.replay_numsamples = num_samples

        np.savez_compressed(file=file, **self.replay)

def pre_figure(figure1,figure2):

    assert figure1.shape == figure2.shape ,'the shape of two figure is different'
    if len(figure1.shape) == 3:
        first_img = figure1.transpose((2, 0, 1))[0]
        second_img = figure2.transpose((2, 0, 1))[0]
    else:
        first_img = figure1
        second_img = figure2

    flow = cv2.calcOpticalFlowFarneback(first_img, second_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

def get_flow_from_inputs_outputs_frame(input_frames,output_frames):
    last_frame = input_frames[-1]
    ouput_frame = output_frames[0]
    batch_size = last_frame.shape[0]
    height = last_frame.shape[1]
    width = last_frame.shape[2]

    flows = np.zeros((batch_size,height,width,2))
    for batch_idx in range(batch_size):
        figure1 = last_frame[batch_idx]
        figure2 = ouput_frame[batch_idx]
        flows[batch_idx,:,:,:,]=pre_figure(figure1[:,:,0],figure2[:,:,0])

    return flows




class MovingMNISTIterator(object):
    def __init__(self,is_CGAN = False):
        self.is_CGAN = is_CGAN
        self.mnist_train_img, self.mnist_train_label,\
        self.mnist_test_img, self.mnist_test_label = load_mnist()

    def get_label_images(self,digitals,width=64,height=64):
        labels = []
        for i in range(10):

            if i in digitals:
                labels.append(np.ones((width,height)))
            else:
                labels.append(np.zeros((width,height)))
        labels = np.array(labels).transpose((1,2,0))
        return labels

    def sample(self, batch_size, in_len,out_len, digitnum=3, width=64, height=64, lower=1.0, upper=3.0,
               index_range=(0, 50000)):
        """

        Parameters
        ----------
        digitnum
        width
        height
        seqlen
        batch_size
        index_range

        Returns
        -------
        seq : np.NDArray
            Shape: (seqlen, batch_size, 1, width, height)
        """
        seqlen = in_len+out_len
        character_indices = np.random.randint(low=index_range[0], high=index_range[1],
                                                 size=(batch_size, digitnum))

        angles = np.random.random((batch_size, digitnum)) * (2 * np.pi)
        magnitudes = np.random.random((batch_size, digitnum)) * (upper - lower) + lower
        velocities = np.zeros((batch_size, digitnum, 2), dtype='float32')
        velocities[..., 0] = magnitudes * np.cos(angles)
        velocities[..., 1] = magnitudes * np.sin(angles)
        xmin = 14.0
        xmax = float(width) - 14.0
        ymin = 14.0
        ymax = float(height) - 14.0
        positions = np.random.uniform(low=xmin, high=xmax,
                                         size=(batch_size, digitnum, 2))
        seq = np.zeros((seqlen, batch_size,height, width,1), dtype='uint8')
        la = []

        for i in range(batch_size):
            if self.is_CGAN:
                digitals = []
            else:
                pass
            for j in range(digitnum):
                ind = character_indices[i, j]
                if self.is_CGAN:
                    digitals.append(self.mnist_train_label[ind])
                else:
                    pass

                v = velocities[i, j, :]
                p = positions[i, j, :]
                img = self.mnist_train_img[ind].reshape((28, 28))
                for k in range(seqlen):
                    topleft_y = int(p[0] - img.shape[0] / 2)
                    topleft_x = int(p[1] - img.shape[1] / 2)
                    seq[k, i, topleft_y:topleft_y + 28, topleft_x:topleft_x + 28,0] = np.maximum(
                        seq[k, i, topleft_y:topleft_y + 28, topleft_x:topleft_x + 28,0],
                        img)
                    v, p = move_step(v, p, [xmin, xmax, ymin, ymax])
            if self.is_CGAN:
                labels_imgs = self.get_label_images(digitals)
                la.append(labels_imgs)
            else:
                pass
        seq = seq.astype(np.float32)
        input_frames = seq[:in_len]
        output_frames = seq[in_len:]

        flows = get_flow_from_inputs_outputs_frame(input_frames,output_frames)
        input_frames = input_frames.transpose((1,0,2,3,4))
        output_frames = output_frames.transpose((1,0,2,3,4))
        input_frames = input_frames / 255.0
        output_frames = output_frames / 255.0
        flows = (flows-np.min(flows))/(np.max(flows)-np.min(flows))

        if self.is_CGAN:
            return input_frames,output_frames,np.array(la)
        else:
            return input_frames, output_frames,flows




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    #set number
    train_batchsize_number = 313*4
    valid_batchsize_number = 63*4
    test_batchsize_number = 94*4
    mnist_generator = MovingMNISTIterator()
    # mnist_generator = MovingMNISTAdvancedIterator(is_CGAN=True)
    batch_size =8
    input_seq_number = 10
    output_seq_number = 10
    input_frames, output_frames, _ = mnist_generator.sample(batch_size,input_seq_number,output_seq_number)
    input_frames = input_frames.transpose((1, 0, 4, 2, 3))
    output_frames = output_frames.transpose((1, 0, 4, 2, 3))
    print(input_frames.shape, output_frames.shape)

    for i in range(train_batchsize_number):
        input_frames, output_frames, _ = mnist_generator.sample(batch_size, input_seq_number, output_seq_number)
        input_frames = input_frames.transpose((1, 0, 4, 2, 3))
        output_frames = output_frames.transpose((1, 0, 4, 2, 3))
        #im = Image.fromarray(input_frames[0, i, :, :, 0] * 255.0)  # numpy 转 image类
        data_path_input = 'E:/data/moving_mnist_data/train_input_' + str(i) + '.npy'
        np.save(data_path_input, input_frames)
        data_path_output = 'E:/data/moving_mnist_data/train_output_' + str(i) + '.npy'
        np.save(data_path_output, output_frames)
        if i%10 == 0:
            print("save train set: [{}/{}]" .format(i+1, train_batchsize_number))

    for i in range(valid_batchsize_number):
        input_frames, output_frames, _ = mnist_generator.sample(batch_size, input_seq_number, output_seq_number)
        input_frames = input_frames.transpose((1, 0, 4, 2, 3))
        output_frames = output_frames.transpose((1, 0, 4, 2, 3))
        #im = Image.fromarray(input_frames[0, i, :, :, 0] * 255.0)  # numpy 转 image类
        data_path_input = 'E:/data/moving_mnist_data/valid_input_' + str(i) + '.npy'
        np.save(data_path_input, input_frames)
        data_path_output = 'E:/data/moving_mnist_data/valid_output_' + str(i) + '.npy'
        np.save(data_path_output, output_frames)
        if i%10 == 0:
            print("save valid set: [{}/{}]" .format(i+1, valid_batchsize_number))

    for i in range(test_batchsize_number):
        input_frames, output_frames, _ = mnist_generator.sample(batch_size, input_seq_number, output_seq_number)
        input_frames = input_frames.transpose((1, 0, 4, 2, 3))
        output_frames = output_frames.transpose((1, 0, 4, 2, 3))
        #im = Image.fromarray(input_frames[0, i, :, :, 0] * 255.0)  # numpy 转 image类
        data_path_input = 'E:/data/moving_mnist_data/test_input_' + str(i) + '.npy'
        np.save(data_path_input, input_frames)
        data_path_output = 'E:/data/moving_mnist_data/test_output_' + str(i) + '.npy'
        np.save(data_path_output, output_frames)
        if i%10 == 0:
            print("save test set: [{}/{}]" .format(i+1, test_batchsize_number))
        #with open(data_path, 'w') as output:
        #print(data_path)
        #im = im.convert('RGB')
        #im.save(data_path)
    #b = np.load("filename.npy")




