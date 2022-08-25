import torch
import numpy as np

def cal_IOU_wFeatureMap(feature_map, vector):
    fm = feature_map
    vector = vector.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    vector = vector.repeat(fm.shape[0],fm.shape[1],fm.shape[2],fm.shape[3],1)

    cat_M_x_GT = torch.cat((vector[...,0:1],vector[...,2:3]), dim=-1)
    cat_M_y_GT = torch.cat((vector[...,1:2],vector[...,3:4]), dim=-1)
    cat_M_x_FM = torch.cat((feature_map[...,0:1],feature_map[...,2:3]), dim=-1)
    cat_M_y_FM = torch.cat((feature_map[...,1:2],feature_map[...,3:4]), dim=-1)

    max_M_x_GT = torch.max(cat_M_x_GT, dim=-1).values.unsqueeze(-1)
    max_M_y_GT = torch.max(cat_M_y_GT, dim=-1).values.unsqueeze(-1)
    max_M_x_FM = torch.max(cat_M_x_FM, dim=-1).values.unsqueeze(-1)
    max_M_y_FM = torch.max(cat_M_y_FM, dim=-1).values.unsqueeze(-1)

    min_M_x_GT = torch.min(cat_M_x_GT, dim=-1).values.unsqueeze(-1)
    min_M_y_GT = torch.min(cat_M_y_GT, dim=-1).values.unsqueeze(-1)
    min_M_x_FM = torch.min(cat_M_x_FM, dim=-1).values.unsqueeze(-1)
    min_M_y_FM = torch.min(cat_M_y_FM, dim=-1).values.unsqueeze(-1)

    minmax_M_x = torch.min(torch.cat((max_M_x_GT, max_M_x_FM), dim=-1), dim=-1).values.unsqueeze(-1)
    maxmin_M_x = torch.max(torch.cat((min_M_x_GT, min_M_x_FM), dim=-1), dim=-1).values.unsqueeze(-1)
    minmax_M_y = torch.min(torch.cat((max_M_y_GT, max_M_y_FM), dim=-1), dim=-1).values.unsqueeze(-1)
    maxmin_M_y = torch.max(torch.cat((min_M_y_GT, min_M_y_FM), dim=-1), dim=-1).values.unsqueeze(-1)

    w_M = torch.clip(torch.min(minmax_M_x-maxmin_M_x, dim=-1).values, 0, 1280).unsqueeze(-1)
    h_M = torch.clip(torch.min(minmax_M_y-maxmin_M_y, dim=-1).values, 0, 1280).unsqueeze(-1)

    inter_M = w_M*h_M

    fm_Area_M = (fm[...,2:3] - fm[...,0:1])*(fm[...,3:4] - fm[...,1:2])
    gt_Area_M = (vector[...,2:3] - vector[...,0:1])*(vector[...,3:4] - vector[...,1:2])
    IoU_M = inter_M/(fm_Area_M+gt_Area_M-inter_M)

    check = IoU_M>=0.5
    indices = check.nonzero()
    return torch.tensor(np.unique(indices[:,2:4].cpu().numpy(),axis=0))


def test():
    Test_gt_vector = torch.Tensor([1, 1, 2, 2])

    Test_feature_map = torch.zeros([1, 3, 160, 160, 4])

    for i in range(160):
        for j in range(160):
            for k in range(3):
                Test_feature_map[0, k, i, j, 0] = i + k
                Test_feature_map[0, k, i, j, 1] = j + k
                Test_feature_map[0, k, i, j, 2] = i + 1 + k
                Test_feature_map[0, k, i, j, 3] = j + 1 + k
    indices = cal_IOU_wFeatureMap(Test_feature_map, Test_gt_vector)
    print(indices)
    print("Test Done")

if __name__ == "__main__":
    test()