import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module


# ============= Model Architecture =============
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups,
                           stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel,
                           groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True,
                                      kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MiniFASNet(Module):
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7),
                 drop_p=0.2, num_classes=3, img_channel=3):
        super(MiniFASNet, self).__init__()
        self.embedding_size = embedding_size

        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])

        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])

        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])

        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]),
              (keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]),
              (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]),
              (keep[33], keep[34]), (keep[36], keep[37])]
        self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = torch.nn.Dropout(p=drop_p)
        self.prob = Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out


# ============= Model Configurations =============
KEEP_DICT = {
    'MiniFASNetV1': [32, 32, 103, 103, 64, 13, 13, 64, 26, 26,
                     64, 13, 13, 64, 52, 52, 64, 231, 231, 128,
                     154, 154, 128, 52, 52, 128, 26, 26, 128, 52,
                     52, 128, 26, 26, 128, 26, 26, 128, 308, 308,
                     128, 26, 26, 128, 26, 26, 128, 512, 512],
    'MiniFASNetV2': [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
                     13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
                     52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
                     128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512]
}


def parse_model_name(model_name):
    """Parse model filename để lấy thông tin"""
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]
    
    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale


def get_kernel(h_input, w_input):
    kernel_size = ((h_input + 15) // 16, (w_input + 15) // 16)
    return kernel_size


def create_model(model_type, h_input, w_input):
    """Tạo model instance"""
    kernel_size = get_kernel(h_input, w_input)
    keep = KEEP_DICT[model_type]
    model = MiniFASNet(keep, embedding_size=128, conv6_kernel=kernel_size,
                       drop_p=0.2, num_classes=3, img_channel=3)
    return model


# ============= Image Processing =============
class CropImage:
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x, y, box_w, box_h = bbox
        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y

        left_top_x = center_x - new_width/2
        left_top_y = center_y - new_height/2
        right_bottom_x = center_x + new_width/2
        right_bottom_y = center_y + new_height/2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1
        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1

        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):
        if not crop:
            return cv2.resize(org_img, (out_w, out_h))
        
        src_h, src_w, _ = org_img.shape
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = \
            self._get_new_box(src_w, src_h, bbox, scale)
        
        img = org_img[left_top_y:right_bottom_y+1, left_top_x:right_bottom_x+1]
        return cv2.resize(img, (out_w, out_h))


def to_tensor(img):
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    return torch.from_numpy(img)


# ============= Main LivenessDetector Class =============
class LivenessDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        self.model_path = model_path
        self.image_cropper = CropImage()
        
        # Parse model info từ filename
        model_name = os.path.basename(model_path)
        self.h_input, self.w_input, self.model_type, self.scale = parse_model_name(model_name)
        
        # Load model
        self.model = create_model(self.model_type, self.h_input, self.w_input)
        self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def _load_weights(self, model_path):
        """Load model weights từ checkpoint"""
        state_dict = torch.load(model_path, map_location=self.device)

        keys = iter(state_dict)
        first_layer_name = next(keys)
        if 'module.' in first_layer_name:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]  # Bỏ 'module.' prefix
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
    
    def predict(self, img, bbox, debug=False):
        param = {
            "org_img": img,
            "bbox": bbox,
            "scale": self.scale if self.scale else 1.0,
            "out_w": self.w_input,
            "out_h": self.h_input,
            "crop": self.scale is not None,
        }
        img_crop = self.image_cropper.crop(**param)

        img_tensor = to_tensor(img_crop)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            result = self.model(img_tensor)
            result = F.softmax(result, dim=1).cpu().numpy()[0]
        
        if debug:
            print(f"Raw output: {result}")
            print(f"Class 0 (Fake): {result[0]:.4f}")
            print(f"Class 1 (Real): {result[1]:.4f}")
            print(f"Class 2 (Unknown): {result[2]:.4f}")

        label_idx = np.argmax(result)
        score = result[label_idx]

        if label_idx == 1:
            label = 'real'
        else:
            label = 'fake'
        
        return score, label
    
    def predict_batch(self, img, bboxes):
        results = []
        for bbox in bboxes:
            score, label = self.predict(img, bbox)
            results.append((score, label))
        return results


# ============= Convenience Functions =============
def load_detector(model_path='models/2.7_80x80_MiniFASNetV2.pth',
                  device='cuda'):
    return LivenessDetector(model_path, device)


# ============= Liveness Detector with Tracking =============
from core.liveness_tracker import LivenessTracker


class LivenessDetectorWithTracking:
    """
    Liveness detector with built-in tracking and voting mechanism
    """
    
    def __init__(self, 
                 model_path,
                 device='cuda',
                 max_history=10,
                 voting_method='weighted',
                 min_samples_for_voting=3,
                 track_timeout=2.0):
        """
        Args:
            model_path: Path to liveness model
            device: 'cuda' or 'cpu'
            max_history: Number of predictions to keep for voting
            voting_method: 'majority', 'weighted', or 'confidence_threshold'
            min_samples_for_voting: Minimum samples needed for reliable voting
            track_timeout: Seconds before a track expires
        """
        self.detector = LivenessDetector(model_path, device)
        self.tracker = LivenessTracker(
            max_history=max_history,
            voting_method=voting_method,
            min_samples_for_voting=min_samples_for_voting,
            track_timeout=track_timeout
        )
        
    def predict_with_tracking(self, img, bboxes, debug=False):
        """
        Predict liveness with tracking and voting
        
        Args:
            img: Input image
            bboxes: List of face bounding boxes
            debug: Print debug info
            
        Returns:
            List of (track_id, bbox, is_real, confidence, vote_info)
        """
        # Get raw predictions
        detections = []
        for bbox in bboxes:
            score, label = self.detector.predict(img, bbox, debug=debug)
            detections.append((bbox, score, label))
        
        # Update tracker and get voted results
        results = self.tracker.update(detections)
        
        if debug:
            print("\n=== Tracking Results ===")
            for track_id, bbox, is_real, confidence, vote_info in results:
                print(f"Track {track_id}: {'REAL' if is_real else 'FAKE'} "
                      f"(conf: {confidence:.3f}, stability: {vote_info.get('stability', 0):.3f}, "
                      f"age: {vote_info.get('age', 0)})")
                print(f"  Vote info: {vote_info}")
        
        return results
    
    def reset_tracking(self):
        """Reset all tracks"""
        self.tracker.reset()
    
    def get_track_info(self, track_id):
        """Get detailed information about a track"""
        return self.tracker.get_track_info(track_id)


def load_detector_with_tracking(
    model_path='models/2.7_80x80_MiniFASNetV2.pth',
    device='cuda',
    max_history=10,
    voting_method='weighted',
    min_samples_for_voting=3
):
    """
    Load liveness detector with tracking
    
    Args:
        model_path: Path to liveness model
        device: 'cuda' or 'cpu'
        max_history: Number of predictions to keep for voting
        voting_method: 'majority', 'weighted', or 'confidence_threshold'
        min_samples_for_voting: Minimum samples needed for reliable voting
    """
    return LivenessDetectorWithTracking(
        model_path=model_path,
        device=device,
        max_history=max_history,
        voting_method=voting_method,
        min_samples_for_voting=min_samples_for_voting
    )
