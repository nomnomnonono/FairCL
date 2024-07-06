import os
import torch
import torchvision.utils as vutils
from helpers import add_scalar_dict


class Trainer:
    def __init__(self):
        pass

    def train_labeled(self, model, train_dataloader, progressbar, writer, args):
        it = 0
        for epoch in range(args.epochs):
            # train with base lr in the first 100 epochs and half the lr in the last 100 epochs
            # lr = args.lr_base / (10 ** (epoch // 100))
            lr = args.lr_base
            model.set_lr(lr)
            writer.add_scalar('LR/learning_rate', lr, it + 1)

            for img_a, att_a in progressbar(train_dataloader):
                model.train()
                
                img_a = img_a.cuda(args.gpu)
                att_a = att_a.cuda(args.gpu).unsqueeze(-1)
                idx = torch.randperm(len(att_a))
                att_b = att_a[idx].contiguous()
                
                att_a = att_a.type(torch.float)
                att_b = att_b.type(torch.float)
                
                att_a_ = (att_a * 2 - 1) * args.thres_int
                if args.b_distribution == 'none':
                    att_b_ = (att_b * 2 - 1) * args.thres_int
                if args.b_distribution == 'uniform':
                    att_b_ = (att_b * 2 - 1) * torch.rand_like(att_b) * (2 * args.thres_int)
                if args.b_distribution == 'truncated_normal':
                    att_b_ = (att_b * 2 - 1) * (torch.fmod(torch.randn_like(att_b), 2) + 2) / 4.0 * (2 * args.thres_int)

                if (it + 1) % (args.n_d + 1) != 0:
                    errD = model.trainD(img_a, att_a, att_a_, att_b, att_b_)
                    add_scalar_dict(writer, errD, it+1, 'D')
                else:
                    errG = model.trainG(img_a, att_a, att_a_, att_b, att_b_)
                    add_scalar_dict(writer, errG, it+1, 'G')
                    progressbar.say(epoch=epoch, iter=it+1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])

                it += 1
            
            if (epoch + 1) % args.sample_interval == 0:
                model.eval()
                with torch.no_grad():
                    samples = [self.fixed_img_a]
                    for i, att_b in enumerate(self.sample_att_b_list):
                        att_b_ = (att_b * 2 - 1) * args.thres_int
                        if i > 0:
                            att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
                        samples.append(model.G(self.fixed_img_a, att_b_))
                    samples = torch.cat(samples, dim=3)
                    writer.add_image('sample', vutils.make_grid(samples, nrow=1, normalize=True, value_range=(-1., 1.)), it+1)
                    vutils.save_image(samples, os.path.join(
                            'output', args.experiment_name, 'sample_training',
                            'Epoch{:d}.jpg'.format(epoch + 1)
                        ), nrow=1, normalize=True, value_range=(-1., 1.))

            if (epoch + 1) % args.save_interval == 0:
                # To save storage space, I only checkpoint the weights of G.
                # If you'd like to keep weights of G, D, optim_G, optim_D, please use save() instead of saveG().
                model.save_model(os.path.join(
                    'output', args.experiment_name, 'checkpoint', 'generator.{:d}.pth'.format(epoch + 1)
                ))


    def train_semi(self, model, train_dataloader, progressbar, writer, args):
        model.set_lr(args.lr_base)
        it = 0
        while True:
            # Update generater train data
            train_dataloader.dataset.mode = "unlabeled"
            indices, labels = [], []
            for img_a, _, index in progressbar(train_dataloader):
                model.eval()
                img_a = img_a.cuda(args.gpu)
                _, logit = model.D(img_a)
                mask = (logit >= args.threshold) | (logit <= (1 - args.threshold))
                indices += [index[i].item() for i in range(len(mask)) if mask[i]]
                labels += [int(logit[i].item() > 0.5) for i in range(len(mask)) if mask[i]]

            if len(indices) == 0:
                print("No more samples to update generater train data")
                break

            train_dataloader.dataset.update_generater_training_data(indices, labels)
            print(f"Update generater train data: {len(indices)} samples")

            train_dataloader.dataset.mode = "gen_semi"
            t_it = 0
            for img_a, att_a, _ in progressbar(train_dataloader):
                model.train()
                
                img_a = img_a.cuda(args.gpu)
                att_a = att_a.cuda(args.gpu).unsqueeze(-1)
                idx = torch.randperm(len(att_a))
                att_b = att_a[idx].contiguous()
                
                att_a = att_a.type(torch.float)
                att_b = att_b.type(torch.float)
                
                att_a_ = (att_a * 2 - 1) * args.thres_int
                if args.b_distribution == 'none':
                    att_b_ = (att_b * 2 - 1) * args.thres_int
                if args.b_distribution == 'uniform':
                    att_b_ = (att_b * 2 - 1) * torch.rand_like(att_b) * (2 * args.thres_int)
                if args.b_distribution == 'truncated_normal':
                    att_b_ = (att_b * 2 - 1) * (torch.fmod(torch.randn_like(att_b), 2) + 2) / 4.0 * (2 * args.thres_int)

                if (t_it + 1) % (args.n_d + 1) != 0:
                    errD = model.trainD(img_a, att_a, att_a_, att_b, att_b_)
                    add_scalar_dict(writer, errD, t_it+1, 'D')
                else:
                    errG = model.trainG(img_a, att_a, att_a_, att_b, att_b_)
                    add_scalar_dict(writer, errG, t_it+1, 'G')
                    progressbar.say(epoch=it+1, iter=t_it+1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])
                t_it += 1
            
            if True:
                model.eval()
                with torch.no_grad():
                    samples = [self.fixed_img_a]
                    for i, att_b in enumerate(self.sample_att_b_list):
                        att_b_ = (att_b * 2 - 1) * args.thres_int
                        if i > 0:
                            att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
                        samples.append(model.G(self.fixed_img_a, att_b_))
                    samples = torch.cat(samples, dim=3)
                    writer.add_image('sample', vutils.make_grid(samples, nrow=1, normalize=True, value_range=(-1., 1.)), it+1)
                    vutils.save_image(samples, os.path.join(
                            'output', args.experiment_name, 'sample_training',
                            'Iter{:d}.jpg'.format(it+1)
                        ), nrow=1, normalize=True, value_range=(-1., 1.))
            
            # Update classifier train data
            train_dataloader.dataset.mode = "gen_semi"
            images, labels = [], []
            for img_a, att_a, index in progressbar(train_dataloader):
                model.eval()
                img_a = img_a.cuda(args.gpu)
                att_a = att_a.cuda(args.gpu).unsqueeze(-1)
                idx = torch.randperm(len(att_a))
                att_b = att_a[idx].contiguous()
                
                att_a = att_a.type(torch.float)
                att_b = att_b.type(torch.float)
                
                att_a_ = (att_a * 2 - 1) * args.thres_int
                if args.b_distribution == 'none':
                    att_b_ = (att_b * 2 - 1) * args.thres_int
                if args.b_distribution == 'uniform':
                    att_b_ = (att_b * 2 - 1) * torch.rand_like(att_b) * (2 * args.thres_int)
                if args.b_distribution == 'truncated_normal':
                    att_b_ = (att_b * 2 - 1) * (torch.fmod(torch.randn_like(att_b), 2) + 2) / 4.0 * (2 * args.thres_int)

                zs_a = model.G(img_a, mode='enc')
                gen_b = model.G(zs_a, att_b_, mode='dec')
                _, logit = model.D(gen_b)
                mask = ((att_a == 1) & (logit <= (1 - args.threshold))) | ((att_a == 0) & (logit >= args.threshold))
                labels += [0 for i in range(len(mask)) if mask[i]]
                images += [gen_b[i].detach().cpu() for i in range(len(mask)) if mask[i]]

            root = os.path.join('output', args.experiment_name, "generated_sample")
            train_dataloader.dataset.update_classifier_training_data(images, labels, root)
            print(f"Update classifier train data: {len(labels)} samples")

            train_dataloader.dataset.mode = "clf_semi"
            t_it = 0
            for img_a, att_a, _ in progressbar(train_dataloader):
                model.train()
                
                img_a = img_a.cuda(args.gpu)
                att_a = att_a.cuda(args.gpu).unsqueeze(-1)
                idx = torch.randperm(len(att_a))
                att_b = att_a[idx].contiguous()
                
                att_a = att_a.type(torch.float)
                att_b = att_b.type(torch.float)
                
                att_a_ = (att_a * 2 - 1) * args.thres_int
                if args.b_distribution == 'none':
                    att_b_ = (att_b * 2 - 1) * args.thres_int
                if args.b_distribution == 'uniform':
                    att_b_ = (att_b * 2 - 1) * torch.rand_like(att_b) * (2 * args.thres_int)
                if args.b_distribution == 'truncated_normal':
                    att_b_ = (att_b * 2 - 1) * (torch.fmod(torch.randn_like(att_b), 2) + 2) / 4.0 * (2 * args.thres_int)

                errD = model.trainD(img_a, att_a, att_a_, att_b, att_b_)
                progressbar.say(epoch=it+1, iter=t_it+1, d_loss=errD['d_loss'])
                t_it += 1

            it += 1

        model.save_model(os.path.join(
            'output', args.experiment_name, 'checkpoint', 'generator.pth'
        ))


    def train(self, model, train_dataloader, progressbar, writer, args):
        pass

    def set_valid_image(self, valid_dataloader, args):
        fixed_img_a, fixed_att_a = next(iter(valid_dataloader))
        self.fixed_img_a = fixed_img_a.cuda(args.gpu)
        fixed_att_a = fixed_att_a.cuda(args.gpu).type(torch.float).unsqueeze(-1)
        self.sample_att_b_list = [fixed_att_a]
        for i in range(args.n_attrs):
            tmp = fixed_att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            #tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
            self.sample_att_b_list.append(tmp)
