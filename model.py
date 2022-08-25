import pytorch_model_summary
import torch

original_model = torch.load('./runs/train/exp53/weights/best.pt')['model'].float().cuda()

pytorch_model_summary.summary(original_model, torch.zeros(1,4,1280,1280).cuda(), show_hierarchical=True,show_input=True, print_summary=True)