import numpy as np
import cv2
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch
from torch import nn

def plot_attention_maps(attns, img, num_heads=12):

    
    fig, ax = plt.subplots(num_heads, 4)
    fig.set_size_inches(18.5, 70)


    # Stack attention maps
    att_maps = torch.stack(attns)
    att_maps = att_maps.squeeze(1)
    # Average the attention weights across all heads.
    #att_mat = torch.mean(att_mat, dim=0)


    for j in range(num_heads):
        #fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        #plt.imsave(fname=fname, arr=attentions[j], format='png')
        #print(f"{fname} saved.")

        # Take the Last attention map
        cur_attention_head = att_maps[j,:,:]
        #Use Classification Token's attention
        cur_attn_map = cur_attention_head[0,1:]

        grid_size = int(np.sqrt(cur_attn_map.shape[0])) # 49, 49
        mask = cur_attn_map.reshape(grid_size, grid_size).detach().numpy()

        im = transforms.ToPILImage()(img.squeeze(0))
        # Maybe use interpolate instead of resize
        
        interpol_mask = nn.functional.interpolate(torch.tensor(mask).unsqueeze(0).unsqueeze(1), scale_factor=32, mode="nearest")
        interpol_mask = interpol_mask[0,0,:,:].unsqueeze(2)
        result_interpol = (torch.tensor(interpol_mask[:,:,0]) * img[0][0])
        
        resize_mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
        result_resize = (torch.tensor(resize_mask[:,:,0]) * img[0][0])


        ax[j][0].imshow(im)
        ax[j][0].set_title("Orginal")
        
        
        ax[j][1].imshow(im)
        #ax[j][1].imshow(mask, cmap='hot', alpha=0.7)
        ax[j][1].imshow(resize_mask, alpha=0.7)
        ax[j][1].set_title("Head Attention" + str(j))
        
        ax[j][2].imshow(result_resize)
        ax[j][2].set_title("Head result_resize Attention" + str(j))

        ax[j][3].imshow(resize_mask)
        ax[j][3].set_title("Head resize_mask Attention" + str(j))

    
    plt.show()