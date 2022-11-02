import torch
import numpy as np


def embedding_mask(x_categ, x_cont, cat_mask, con_mask,model, type= None):
    '''
        x_categ 256 * 9
        x_cont 256 * 6
        cat_mask 256 * 9
        con_mask 256 * 6
    '''
    # print(x_categ)
    # print(x_cont)
    device = x_cont.device
    offset = model.categories_offset.type_as(x_categ)
    x_categ = x_categ + offset
    x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2,model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:]= model.simple_MLP[i](x_cont[:,i])

    x_cont_enc = x_cont_enc.to(device)

    ## mask with corresponding
    if "mask_multi" in type:
        con_mask_token = torch.arange(1, x_cont.shape[1] + 1)
        con_mask_token = con_mask_token.repeat(x_cont.shape[0], 1)
        con_mask_token = con_mask_token.to(device)

        cat_mask_token = torch.arange(1, x_categ.shape[1] + 1)
        cat_mask_token = cat_mask_token.repeat(x_categ.shape[0], 1)
        cat_mask_token = cat_mask_token.to(device)
    ## mask with one
    elif "mask_one" in type:
        con_mask_token = torch.ones_like(con_mask)
        cat_mask_token = torch.ones_like(cat_mask)

    cat_mask_temp = model.cat_mask(cat_mask_token)
    con_mask_temp = model.con_mask(con_mask_token)

    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    return x_categ, x_categ_enc, x_cont_enc

def embed_data_mask_train(x_categ, x_cont, cat_mask, con_mask,model):
    '''
    x_categ 256 * 9
    x_cont 256 * 6
    cat_mask 256 * 9
    con_mask 256 * 6
    '''

    device = x_cont.device


    offset = model.categories_offset.type_as(x_categ)
    x_categ = x_categ + offset
    x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape


    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    else:
        raise Exception('This case should not work!')


    x_cont_enc = x_cont_enc.to(device)

    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)

    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]



    return x_categ, x_categ_enc, x_cont_enc




