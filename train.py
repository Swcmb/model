import random  # 随机数控制（用于批次级对抗种子派生）
import json  # 用于保存与加载对抗配置
import os  # 文件保存路径
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库（当前文件中可能未使用）
import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
from torch_geometric.data import Data  # 图数据结构支持
from layer import apply_augmentation, adversarial_step_multi  # 增强与对抗步骤
from log_output_manager import save_result_text, get_run_paths  # 日志与结果管理
from sklearn.metrics import roc_auc_score,roc_curve,average_precision_score,f1_score,auc,precision_score,recall_score,confusion_matrix
# 可视化：按Epoch绘制 train_loss / val_loss / val_AUROC
from visualization import load_epoch_metrics_csv, plot_epoch_curves_from_df


def train_model(model, optimizer, data_o, data_a, train_loader, test_loader, args, fold_idx=None):  # 定义主训练函数，增加fold索引
    m = torch.nn.Sigmoid()  # 实例化Sigmoid函数，用于将模型输出转换为概率
    loss_fct = torch.nn.BCELoss()  # 实例化二元交叉熵损失函数（用于主任务）
    b_xent = nn.BCEWithLogitsLoss()  # 实例化带Logits的二元交叉熵损失，更稳定（用于对比和对抗损失）
    ce_loss = nn.CrossEntropyLoss()  # 用于 MoCo InfoNCE
    node_loss = nn.BCEWithLogitsLoss()  # 同上，用于节点级别的对抗损失
    loss_history = []  # 创建一个列表来记录每个批次的损失值

    model.to('cuda')  # 固定使用GPU
    data_o.to('cuda')  # 固定使用GPU
    data_a.to('cuda')  # 固定使用GPU
    device = 'cuda'
    aug_mode = getattr(args, 'augment_mode', 'static')
    # 在线增强阶段固定为 random_permute_features（不再取列表首个）
    aug_online = "random_permute_features"
    noise_std = float(getattr(args, 'noise_std', 0.01) or 0.01)
    mask_rate = float(getattr(args, 'mask_rate', 0.1) or 0.1)
    base_seed = getattr(args, 'augment_seed', None)
    if base_seed is None:
        base_seed = int(getattr(args, 'seed', 0))

    # 保存当次训练（该折）的对抗配置到 metrics/adv_config_<fold>.json，便于跨折与跨次对比复现
    try:
        run_id = (get_run_paths().get('run_id') or '')
        fold_tag = f"fold_{fold_idx}" if fold_idx is not None else "fold"
        # ✅ 修复2：统一使用 derive_adv_seed 函数派生种子
        from utils import derive_adv_seed
        base_adv_seed = derive_adv_seed(args, fold_idx or 0, 0, 0)

        adv_cfg = {
            "mode": str(getattr(args, "adv_mode", "none")),
            "norm": str(getattr(args, "adv_norm", "linf")),
            "eps": float(getattr(args, "adv_eps", 0.01) or 0.01),
            "alpha": float(getattr(args, "adv_alpha", 0.005) or 0.005),
            "steps": int(getattr(args, "adv_steps", 0) or 0),
            "rand_init": bool(getattr(args, "adv_rand_init", False)),
            "project": bool(getattr(args, "adv_project", True)),
            "agg": str(getattr(args, "adv_agg", "mean")),
            "budget": str(getattr(args, "adv_budget", "independent")),
            "use_amp": bool(getattr(args, "adv_use_amp", False)),
            "on_moco": bool(getattr(args, "adv_on_moco", False)),
            "clip": [
                float(getattr(args, "adv_clip_min", float("-inf"))),
                float(getattr(args, "adv_clip_max", float("inf")))
            ],
            "seed": {
                "base_seed": int(getattr(args, "seed", 0)),
                "base_adv_seed": int(base_adv_seed),
                "derive_rule": "seed_batch = base_adv_seed + epoch*1000 + iter"
            },
            "views": {
                "attack_o": True,
                "attack_augmented": bool(getattr(args, "adv_on_moco", False)),
                "delta": "independent_per_view"
            }
        }
        fname = f"adv_config_{fold_tag}_{run_id}.json" if run_id else f"adv_config_{fold_tag}.json"
        save_result_text(json.dumps(adv_cfg, ensure_ascii=False, indent=2), filename=fname, subdir="metrics")
        if run_id:
            print(f"[SAVE] adv config saved: metrics/{fname}")
        else:
            print(f"[SAVE] adv config saved: metrics/{fname} (no run_id)")
    except Exception as _e:
        print(f"[SAVE] Failed to write adv config: {_e}")

    # Train model  # 注释：训练模型
    lbl = data_a.y  # 获取对抗数据的标签（用于对比学习）
    print('Start Training...')  # 打印开始训练的信息
    # 记录每个epoch的训练指标（便于保存）
    epoch_metrics = []  # 每项为 dict：{'epoch':i, 'auroc':.., 'auprc':.., 'precision':.., 'recall':.., 'f1':.., 'cm':(tn,fp,fn,tp)}

    for epoch in range(args.epochs):  # 开始按设定的轮数进行训练循环
        print('-------- Epoch ' + str(epoch + 1) + ' --------')  # 打印当前轮数
        y_pred_train = []  # 初始化列表，用于存储当前轮次的预测值
        y_label_train = []  # 初始化列表，用于存储当前轮次的真实标签
        loss_train = torch.tensor(0.0) # 初始化训练损失，防止在训练加载器为空时引用错误
        # 初始化分项损失，避免空训练集时报未绑定
        loss1 = torch.tensor(0.0, device=device)
        loss2 = torch.tensor(0.0, device=device)
        loss3 = torch.tensor(0.0, device=device)

        # 为节点级别的对抗损失创建标签（动态节点数 + 设备对齐）
        n_nodes = int(data_o.x.size(0))
        lbl_1 = torch.ones(1, n_nodes, device=device)
        lbl_2 = torch.zeros(1, n_nodes, device=device)
        lbl2 = torch.cat((lbl_1, lbl_2), 1)

        for i, (label, inp) in enumerate(train_loader):  # 遍历训练数据加载器，获取每个批次的标签和输入

            label = label.to('cuda')  # 固定使用GPU
            # 异常防护：空 batch（如早期调试/数据问题）直接跳过，避免产生 nan
            if label.numel() == 0:
                continue

            # 在线增强：每个 batch 动态生成 data_a_aug
            if aug_mode == 'online':
                # 为每个 batch 派生稳定种子：seed + epoch*1000 + iter
                seed_batch = int(base_seed) + epoch * 1000 + i
                # apply_augmentation 支持 torch.Tensor；返回 CPU 张量，需移回原 device
                # 视图生成起止打印（在线增强）
                if i == 0:
                    try:
                        print(f"[AUG][online][epoch={epoch+1}][iter={i}] start name={aug_online} noise_std={noise_std} mask_rate={mask_rate} seed={seed_batch}")
                    except Exception:
                        pass
                aug_x = apply_augmentation(
                    aug_online,
                    data_a.x,  # 可直接传 torch.Tensor
                    noise_std=noise_std,
                    mask_rate=mask_rate,
                    seed=seed_batch
                )
                if i == 0:
                    try:
                        _shape = tuple(aug_x.shape) if hasattr(aug_x, "shape") else "-"
                        print(f"[AUG][online][epoch={epoch+1}][iter={i}] done name={aug_online} shape={_shape}")
                    except Exception:
                        pass
                if isinstance(aug_x, torch.Tensor):
                    aug_x = aug_x.to(device)
                else:
                    aug_x = torch.tensor(aug_x, dtype=data_a.x.dtype, device=device)
                # 构造仅替换 x 的临时 Data；共享 edge_index 与 y
                data_a_aug = Data(x=aug_x, y=data_a.y, edge_index=data_a.edge_index)
                if i == 0:
                    print(f"[ONLINE-AUG] mode=online name={aug_online} noise_std={noise_std} mask_rate={mask_rate} base_seed={base_seed}")
            else:
                data_a_aug = data_a

            model.train()  # 将模型设置为训练模式
            optimizer.zero_grad()  # 清除上一批次的梯度

            # 仅在训练后期启用 PGD：epoch+1 >= adv_warmup_end
            if str(getattr(args, "adv_mode", "none")) == "mgraph" and int(epoch + 1) >= int(getattr(args, "adv_warmup_end", 0) or 0):
                # 多图对抗：构造闭包并生成对抗后的 X
                use_moco_adv = bool(getattr(args, "adv_on_moco", False))
                # 准备输入列表：总是包含 data_o.x；当 use_moco_adv 为真时，增加增强视图 x
                _X_list = [data_o.x] + ([data_a_aug.x] if use_moco_adv else [])

                def _adv_loss_fn(perturbed_list):
                    # perturbed_list 对应 _X_list 的顺序
                    xo = perturbed_list[0]
                    xa = perturbed_list[1] if (use_moco_adv and len(perturbed_list) > 1) else data_a_aug.x
                    data_o_adv = Data(x=xo, y=data_o.y, edge_index=data_o.edge_index)
                    data_a_use = Data(x=xa, y=data_a_aug.y, edge_index=data_a_aug.edge_index)
                    out, cos, cos_a, _, lgts, lg1 = model(data_o_adv, data_a_use, inp)

                    lg = torch.squeeze(m(out))
                    l1 = loss_fct(lg, label.float())
                    if float(getattr(args, "loss_ratio2", 0.0) or 0.0) > 0.0:
                        if isinstance(cos, (list, tuple)):
                            _losses = [ce_loss(lv, tv) for lv, tv in zip(cos, cos_a)]
                            l2 = torch.stack(_losses).mean()
                        else:
                            l2 = ce_loss(cos, cos_a)
                    else:
                        l2 = torch.tensor(0.0, device=device)
                    l3 = node_loss(lgts, lbl2.float())
                    return args.loss_ratio1 * l1 + args.loss_ratio2 * l2 + args.loss_ratio3 * l3

                # ✅ 修复2：在调用对抗生成前，统一使用 derive_adv_seed 派生种子
                from utils import derive_adv_seed
                seed_batch = derive_adv_seed(args, fold_idx or 0, epoch, i)
                try:
                    torch.manual_seed(seed_batch)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_batch)
                except Exception:
                    pass
                try:
                    np.random.seed(seed_batch)
                    random.seed(seed_batch)
                except Exception:
                    pass

                # 生成对抗样本（遵循 adv_budget=independent、各视图独立 delta、clamp 请通过 CLI 设置）
                X_adv_list = adversarial_step_multi(_X_list, _adv_loss_fn, args)
                xo_adv = X_adv_list[0]
                xa_adv = X_adv_list[1] if (use_moco_adv and len(X_adv_list) > 1) else data_a_aug.x

                # 最终一次前向与损失（使用对抗后的特征）
                data_o_use = Data(x=xo_adv, y=data_o.y, edge_index=data_o.edge_index)
                data_a_use = Data(x=xa_adv, y=data_a_aug.y, edge_index=data_a_aug.edge_index)
                output, cla_os, cla_os_a, _, logits, log1 = model(data_o_use, data_a_use, inp)

                log = torch.squeeze(m(output))
                loss1 = loss_fct(log, label.float())
                if float(getattr(args, "loss_ratio2", 0.0) or 0.0) > 0.0:
                    if isinstance(cla_os, (list, tuple)):
                        losses = [ce_loss(lg, tg) for lg, tg in zip(cla_os, cla_os_a)]
                        loss2 = torch.stack(losses).mean()
                    else:
                        loss2 = ce_loss(cla_os, cla_os_a)
                else:
                    loss2 = torch.tensor(0.0, device=device)
                loss3 = node_loss(logits, lbl2.float())
                loss_train = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3

                if i == 0 and epoch == 0:
                    print(f"[ADV] mode=mgraph budget={getattr(args,'adv_budget','independent')} "
                          f"norm={getattr(args,'adv_norm','linf')} eps={getattr(args,'adv_eps',0.01)} "
                          f"alpha={getattr(args,'adv_alpha',0.005)} steps={getattr(args,'adv_steps',0)} "
                          f"on_moco={use_moco_adv} clamp=[{getattr(args,'adv_clip_min',float('-inf'))},"
                          f"{getattr(args,'adv_clip_max',float('inf'))}]")
            else:
                # 保持原有“干净输入”路径
                output, cla_os, cla_os_a, _, logits, log1 = model(data_o, data_a_aug, inp)  # 将数据输入模型，获取多个输出

                log = torch.squeeze(m(output))  # 对主任务输出应用Sigmoid并压缩维度
                loss1 = loss_fct(log, label.float())  # 计算主任务的二元交叉熵损失
                # MoCo：支持单/多视图；当 alpha=0 时跳过计算以隔离监督路径
                if float(getattr(args, "loss_ratio2", 0.0) or 0.0) > 0.0:
                    if isinstance(cla_os, (list, tuple)):
                        losses = [ce_loss(lg, tg) for lg, tg in zip(cla_os, cla_os_a)]
                        loss2 = torch.stack(losses).mean()
                    else:
                        loss2 = ce_loss(cla_os, cla_os_a)
                else:
                    loss2 = torch.tensor(0.0, device=device)
                # 节点级对抗损失改为 loss_ratio3 对应
                loss3 = node_loss(logits, lbl2.float())
                # 总损失仅使用 1:监督、2:对比、3:节点对抗
                loss_train = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3
            # print("loss_train: ",loss_train)  # 被注释掉的调试语句

            loss_history.append(loss_train.item())  # 记录当前批次的总损失
            loss_train.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

            label_ids = label.to('cpu').numpy()  # 将标签移回CPU并转为numpy数组
            y_label_train = y_label_train + label_ids.flatten().tolist()  # 收集真实标签
            y_pred_train = y_pred_train + log.flatten().tolist()  # 收集预测概率

            if i % 100 == 0:  # 每100个批次
                # 打印当前轮次、迭代次数和训练损失（含分项：task/contrastive/adversarial）
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()) + ' | task_loss:' + format(loss1.item(), '.4f') + ' cont_loss:' + format(loss2.item(), '.4f') + ' adv_loss:' + format(loss3.item(), '.4f'))

        # 在训练集非空时计算各项指标，否则默认0，避免程序出错
        if y_label_train:
            roc_train = roc_auc_score(y_label_train, y_pred_train)
            auprc_train = average_precision_score(y_label_train, y_pred_train)
            outputs_train = np.asarray((np.asarray(y_pred_train) >= 0.5).astype(int))
            precision_train = precision_score(y_label_train, outputs_train, zero_division="warn")
            recall_train = recall_score(y_label_train, outputs_train, zero_division="warn")
            f1_train = f1_score(y_label_train, outputs_train, zero_division="warn")
            tn, fp, fn, tp = confusion_matrix(y_label_train, outputs_train).ravel()
        else:
            roc_train = 0.0
            auprc_train = 0.0
            precision_train = 0.0
            recall_train = 0.0
            f1_train = 0.0
            tn = fp = fn = tp = 0

        # 保存到本地列表以便写入CSV（含验证集评估）
        # 进行一次验证集评估，得到每个epoch的 val_loss 与 val_auroc
        try:
            val_auroc_ep, _, _, _, _, val_loss_tensor, _ = test(model, test_loader, data_o, data_a, args)
            val_loss_ep = float(val_loss_tensor.item()) if hasattr(val_loss_tensor, "item") else float(val_loss_tensor)
        except Exception:
            # 兜底，确保 val_loss 不缺省
            val_auroc_ep = 0.0
            try:
                val_loss_ep = float(loss_train.item())
            except Exception:
                val_loss_ep = 0.0

        epoch_metrics.append({
            'epoch': epoch + 1,
            # 训练集指标
            'auroc': roc_train,
            'auprc': auprc_train,
            'precision': precision_train,
            'recall': recall_train,
            'f1': f1_train,
            'cm': (tn, fp, fn, tp),
            'task_loss': float(loss1.item()),
            'cont_loss': float(loss2.item()),
            'adv_loss': float(loss3.item()),
            'loss_train': float(loss_train.item()),
            # 验证集指标
            'val_loss': (float(val_loss_ep) if val_loss_ep is not None else None),
            'val_auroc': (float(val_auroc_ep) if val_auroc_ep is not None else None)
        })

        # 打印当前轮次的总结信息
        print('epoch: {:04d}'.format(epoch + 1),'loss_train: {:.4f}'.format(loss_train.item()),
                'task_loss: {:.4f}'.format(loss1.item()), 'cont_loss: {:.4f}'.format(loss2.item()),
                'adv_loss: {:.4f}'.format(loss3.item()), 'auroc_train: {:.4f}'.format(roc_train),
                'auprc_train: {:.4f}'.format(auprc_train), 'precision_train: {:.4f}'.format(precision_train),
                'recall_train: {:.4f}'.format(recall_train), 'f1_train: {:.4f}'.format(f1_train),
                f'cm_train: (tn={tn}, fp={fp}, fn={fn}, tp={tp})')

        if hasattr(torch.cuda, 'empty_cache'):  # 如果PyTorch版本支持
            torch.cuda.empty_cache()  # 清空GPU缓存，释放不必要的显存
    print("Optimization Finished!")  # 所有轮次训练完成后，打印优化完成

    # 将每epoch训练指标写入 EM/result 当前运行目录下的CSV
    try:
        run_id = (get_run_paths().get('run_id') or '')
        fold_tag = f"fold_{fold_idx}" if fold_idx is not None else "fold"
        fname = f"train_epoch_metrics_{fold_tag}_{run_id}.csv" if run_id else f"train_epoch_metrics_{fold_tag}.csv"
        # 构造CSV文本
        lines = ["epoch,loss_train,val_loss,task_loss,cont_loss,adv_loss,auroc,val_auroc,auprc,precision,recall,f1,tn,fp,fn,tp"]
        for em in epoch_metrics:
            tn, fp, fn, tp = em['cm']
            # 强制写数值，确保 val_loss 不缺省
            val_loss_val = em.get('val_loss')
            if val_loss_val is None:
                val_loss_val = em.get('loss_train', 0.0)
            val_auroc_val = em.get('val_auroc')
            if val_auroc_val is None:
                val_auroc_val = 0.0
            val_loss_str = f"{float(val_loss_val):.6f}"
            val_auroc_str = f"{float(val_auroc_val):.6f}"
            lines.append("{epoch},{loss:.6f},{val_loss},{tl:.6f},{cl:.6f},{al:.6f},{auc:.6f},{val_auc},{auprc:.6f},{prec:.6f},{rec:.6f},{f1:.6f},{tn},{fp},{fn},{tp}".format(
                epoch=em['epoch'],
                loss=em['loss_train'],
                val_loss=val_loss_str,
                tl=em['task_loss'],
                cl=em['cont_loss'],
                al=em['adv_loss'],
                auc=em['auroc'],            # 训练AUROC（保持历史）
                val_auc=val_auroc_str,      # 验证AUROC
                auprc=em['auprc'],
                prec=em['precision'],
                rec=em['recall'],
                f1=em['f1'],
                tn=tn, fp=fp, fn=fn, tp=tp
            ))
        save_result_text("\n".join(lines), filename=fname, subdir="metrics")
        print(f"[SAVE] Per-epoch train metrics saved: {fname}")
        # 自动生成按Epoch的训练/验证损失与验证AUROC三曲线图（双y轴）
        try:
            # 直接使用内存中的 epoch_metrics 构建 DataFrame 绘图，避免因路径或命名不一致导致漏图
            import pandas as _pd
            df = _pd.DataFrame(epoch_metrics)
            # 列要求：epoch、loss_train、val_loss、val_auroc 均存在；val_loss 必不可缺
            if not {"epoch","loss_train","val_loss"}.issubset(set(df.columns)):
                raise RuntimeError("epoch_metrics 列缺失，无法绘制三曲线")
            save_png = f"epoch_curves_{fold_tag}.png"
            plot_epoch_curves_from_df(df, save_path=save_png, smooth=None)
            print(f"[SAVE] Epoch curves figure saved: {save_png} (redirected to figure/)")
        except Exception as _e_vis:
            print(f"[VIS] Failed to plot epoch curves: {_e_vis}")
    except Exception as _e:
        print(f"[SAVE] Failed to write per-epoch metrics: {_e}")

    # Testing  # 注释：测试阶段
    # 在进入测试前注入当前折信息（用于文件命名）
    try:
        setattr(args, "_current_fold", fold_idx if 'fold_idx' in locals() else getattr(args, "_current_fold", None))
    except Exception:
        pass
    # 调用test函数，在测试集上评估最终模型
    auroc_test, prc_test, precision_test, recall_test, f1_test, loss_test, cm_test = test(model, test_loader, data_o, data_a, args)
    tn_t, fp_t, fn_t, tp_t = cm_test
    # 打印测试集上的各项性能指标
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auroc_test: {:.4f}'.format(auroc_test),
          'auprc_test: {:.4f}'.format(prc_test), 'precision_test: {:.4f}'.format(precision_test),
          'recall_test: {:.4f}'.format(recall_test), 'f1_test: {:.4f}'.format(f1_test),
          f'cm_test: (tn={tn_t}, fp={fp_t}, fn={fn_t}, tp={tp_t})')
    
    # 返回测试结果（含混淆矩阵）
    return {
        'auroc': auroc_test,
        'auprc': prc_test,
        'precision': precision_test,
        'recall': recall_test,
        'f1': f1_test,
        'loss': loss_test.item(),
        'cm': (int(tn_t), int(fp_t), int(fn_t), int(tp_t))
    }


def test(model, loader, data_o, data_a, args):  # 定义测试函数

    m = torch.nn.Sigmoid()  # 实例化Sigmoid
    loss_fct = torch.nn.BCELoss()  # 实例化损失函数
    b_xent = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    node_loss = nn.BCEWithLogitsLoss()


    model.eval()  # 将模型设置为评估模式（会关闭dropout等）
    y_pred = []  # 初始化列表，用于存储预测值
    y_pred_logits = []  # 原始未Sigmoid的logits（用于温度校准）
    y_label = []  # 初始化列表，用于存储真实标签
    loss = torch.tensor(0.0) # 初始化损失，防止在加载器为空时引用错误
    lbl = data_a.y  # 获取对抗数据的标签

    # 同样为对抗损失创建标签（动态节点数 + 设备对齐）
    device = 'cuda'
    n_nodes = int(data_o.x.size(0))
    lbl_1 = torch.ones(1, n_nodes, device=device)
    lbl_2 = torch.zeros(1, n_nodes, device=device)
    lbl2 = torch.cat((lbl_1, lbl_2), 1)

    with torch.no_grad():  # 在此代码块中，不计算梯度，以节省计算资源
        for i, (label, inp) in enumerate(loader):  # 遍历测试数据加载器

            label = label.to('cuda')  # 固定使用GPU
            # 异常防护：空 batch（如早期调试/数据问题）直接跳过，避免产生 nan
            if label.numel() == 0:
                continue

            # 测试阶段不进行在线增强，保持 data_a 静态
            output, cla_os, cla_os_a, _, logits, log1 = model(data_o, data_a, inp)  # 前向传播
            # 原始logit与Sigmoid概率
            logit_raw = torch.squeeze(output)
            log = torch.squeeze(m(output))  # 获取主任务预测概率

            # 计算测试集上的损失（尽管在测试阶段通常更关心指标而非损失值）
            loss1 = loss_fct(log, label.float())
            if float(getattr(args, "loss_ratio2", 0.0) or 0.0) > 0.0:
                if isinstance(cla_os, (list, tuple)):
                    losses = [ce_loss(lg, tg) for lg, tg in zip(cla_os, cla_os_a)]
                    loss2 = torch.stack(losses).mean()
                else:
                    loss2 = ce_loss(cla_os, cla_os_a)
            else:
                loss2 = torch.tensor(0.0, device=device)
            # 节点级对抗损失改为 loss_ratio3 对应
            loss3 = node_loss(logits, lbl2.float())
            loss = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3

            label_ids = label.to('cpu').numpy()  # 将标签移回CPU
            y_label = y_label + label_ids.flatten().tolist()  # 收集真实标签
            y_pred = y_pred + log.flatten().tolist()  # 收集预测概率
            y_pred_logits = y_pred_logits + logit_raw.flatten().tolist()  # 收集原始logits
    
    # 如果测试集为空，则返回0，避免程序崩溃
    if not y_label:
        # 与正常分支保持一致返回项数：auroc, auprc, precision, recall, f1, loss, cm
        return 0.0, 0.0, 0.0, 0.0, 0.0, loss, (0, 0, 0, 0)

    # 阶段A：温度校准与阈值扫描（不改变原返回，结果另存）
    try:
        do_scan = bool(getattr(args, "enable_threshold_scan", False))
        do_temp = bool(getattr(args, "enable_temp_scaling", False))
        tmin = float(getattr(args, "threshold_min", 0.35))
        tmax = float(getattr(args, "threshold_max", 0.65))
        tstep = float(getattr(args, "threshold_step", 0.01))
        Tmin = float(getattr(args, "temp_grid_min", 0.5))
        Tmax = float(getattr(args, "temp_grid_max", 3.0))
        Tnum = int(getattr(args, "temp_grid_num", 26))

        best_t = None
        best_f1 = None
        best_t_cal = None
        best_f1_cal = None
        best_T = None

        y_true_np = np.asarray(y_label, dtype=np.int64)
        probs_np = np.asarray(y_pred, dtype=np.float32)

        # 简易温度校准（网格搜索）：最小化 BCE
        if do_temp and len(y_pred_logits) == len(y_label):
            logits_np = np.asarray(y_pred_logits, dtype=np.float32)
            T_candidates = np.linspace(Tmin, Tmax, num=max(2, Tnum))
            bce_min = None
            T_opt = None
            for T in T_candidates:
                # 数值稳定：裁剪 exp 的输入，避免溢出
                z = -logits_np / float(T)
                z = np.clip(z, -60.0, 60.0)
                probs_T = 1.0 / (1.0 + np.exp(z))
                # 避免log(0)
                eps = 1e-7
                probs_T = np.clip(probs_T, eps, 1.0 - eps)
                # 二分类交叉熵
                bce = -(y_true_np * np.log(probs_T) + (1 - y_true_np) * np.log(1.0 - probs_T)).mean()
                if (bce_min is None) or (bce < bce_min):
                    bce_min = bce
                    T_opt = float(T)
            best_T = T_opt

        # 阈值扫描（原概率）
        if do_scan:
            ths = np.arange(tmin, tmax + 1e-12, tstep)
            def _f1_at_thresh(p, thr):
                preds = (p >= thr).astype(np.int64)
                from sklearn.metrics import f1_score
                return f1_score(y_true_np, preds, zero_division="warn")
            f1_vals = [ _f1_at_thresh(probs_np, thr) for thr in ths ]
            idx = int(np.argmax(f1_vals))
            best_t, best_f1 = float(ths[idx]), float(f1_vals[idx])

        # 阈值扫描（温度校准概率）
        if do_scan and do_temp and best_T is not None and len(y_pred_logits) == len(y_label):
            probs_cal = 1.0 / (1.0 + np.exp(-np.asarray(y_pred_logits, dtype=np.float32) / float(best_T)))
            f1_vals_cal = [ _f1_at_thresh(probs_cal, thr) for thr in ths ]
            idxc = int(np.argmax(f1_vals_cal))
            best_t_cal, best_f1_cal = float(ths[idxc]), float(f1_vals_cal[idxc])

        # 保存结果到 metrics/threshold_scan_*.txt
        try:
            from log_output_manager import save_result_text, get_run_paths
            _paths = get_run_paths()
            _run_id = _paths.get("run_id") or ""
            _fold = getattr(args, "_current_fold", None)
            fname = f"threshold_scan_fold_{_fold}_{_run_id}.txt" if _run_id and _fold else ("threshold_scan.txt" if not _fold else f"threshold_scan_fold_{_fold}.txt")
            lines = []
            lines.append("Threshold Scan & Temperature Scaling")
            lines.append(f"scan_range=[{tmin},{tmax}] step={tstep} temp_grid=[{Tmin},{Tmax}] num={Tnum}")
            if best_t is not None:
                lines.append(f"best_threshold={best_t:.3f} F1@best={best_f1:.4f}")
            if best_T is not None:
                lines.append(f"best_temperature={best_T:.3f}")
            if best_t_cal is not None:
                lines.append(f"calibrated_best_threshold={best_t_cal:.3f} F1_calib@best={best_f1_cal:.4f}")
            save_result_text("\n".join(lines), filename=fname, subdir="metrics")
        except Exception as _e:
            print(f"[SCAN] save failed: {_e}")
    except Exception as _e:
        print(f"[SCAN] skipped due to: {_e}")

    # 在循环结束后，根据所有批次的预测概率计算硬预测（0或1）
    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    # 计算全量测试指标
    auroc = roc_auc_score(y_label, y_pred)
    auprc = average_precision_score(y_label, y_pred)
    precision = precision_score(y_label, outputs, zero_division="warn")
    recall = recall_score(y_label, outputs, zero_division="warn")
    f1 = f1_score(y_label, outputs, zero_division="warn")
    tn, fp, fn, tp = confusion_matrix(y_label, outputs).ravel()

    # 返回测试集上的 AUROC, AUPRC, Precision, Recall, F1, 平均损失 与 混淆矩阵
    # 同时将必要的曲线数据保存到 OUTPUT/result/metrics，便于可视化模块生成图像
    try:
        from log_output_manager import get_run_paths, make_result_run_dir
        _paths = get_run_paths()
        _run_id = _paths.get("run_id") or ""
        _run_result_dir = _paths.get("run_result_dir")
        if not _run_result_dir:
            _run_result_dir = str(make_result_run_dir("data"))
        out_dir = os.path.join(_run_result_dir, "metrics")
        os.makedirs(out_dir, exist_ok=True)
        fold = getattr(args, "_current_fold", None)
        fold_tag = f"fold_{fold}" if fold else "fold"

        # 1) 保存 y_true / y_prob / logits
        arr_path = os.path.join(out_dir, f"y_true_pred_{fold_tag}_{_run_id}.csv" if _run_id else f"y_true_pred_{fold_tag}.csv")
        with open(arr_path, "w", encoding="utf-8") as f:
            f.write("y_true,y_prob,logit\n")
            for yt, yp, lg in zip(y_label, y_pred, y_pred_logits):
                f.write(f"{int(yt)},{float(yp):.8f},{float(lg):.8f}\n")

        # 2) 阈值扫描原始数组（若启用）
        try:
            do_scan = bool(getattr(args, "enable_threshold_scan", False))
            do_temp = bool(getattr(args, "enable_temp_scaling", False))
            if do_scan:
                # 重新构建 ths 与 f1 序列（与上文一致）
                tmin = float(getattr(args, "threshold_min", 0.35))
                tmax = float(getattr(args, "threshold_max", 0.65))
                tstep = float(getattr(args, "threshold_step", 0.01))
                ths = np.arange(tmin, tmax + 1e-12, tstep)
                y_true_np = np.asarray(y_label, dtype=np.int64)
                probs_np = np.asarray(y_pred, dtype=np.float32)

                def _f1_at_thresh(p, thr):
                    preds = (p >= thr).astype(np.int64)
                    from sklearn.metrics import f1_score
                    return f1_score(y_true_np, preds, zero_division="warn")

                f1_vals = [ _f1_at_thresh(probs_np, thr) for thr in ths ]
                th_out = os.path.join(out_dir, f"threshold_scan_{fold_tag}_{_run_id}.csv" if _run_id else f"threshold_scan_{fold_tag}.csv")
                with open(th_out, "w", encoding="utf-8") as f:
                    f.write("threshold,f1\n")
                    for t, fv in zip(ths.tolist(), f1_vals):
                        f.write(f"{t:.6f},{fv:.6f}\n")

                # 温度校准后的阈值扫描（若启用且 best_T 有效）
                # 复用上文已计算的 best_T（若未计算则为 None）
                try:
                    # best_T 在上文温度网格搜索块内赋值；此处读取本地变量
                    best_T_local = locals().get("best_T", None)
                    if do_temp and (best_T_local is not None) and len(y_pred_logits) == len(y_label):
                        logits_np = np.asarray(y_pred_logits, dtype=np.float32)
                        # 数值稳定：裁剪 exp 的输入，避免溢出
                        z = -logits_np / float(best_T_local)
                        z = np.clip(z, -60.0, 60.0)
                        probs_cal = 1.0 / (1.0 + np.exp(z))
                        f1_vals_cal = [ _f1_at_thresh(probs_cal, thr) for thr in ths ]
                        th_cal_out = os.path.join(out_dir, f"threshold_scan_calibrated_{fold_tag}_{_run_id}.csv" if _run_id else f"threshold_scan_calibrated_{fold_tag}.csv")
                        with open(th_cal_out, "w", encoding="utf-8") as f:
                            f.write("threshold,f1_cal\n")
                            for t, fv in zip(ths.tolist(), f1_vals_cal):
                                f.write(f"{t:.6f},{fv:.6f}\n")
                        # 同时保存最佳温度
                        T_json = os.path.join(out_dir, f"temperature_{fold_tag}_{_run_id}.json" if _run_id else f"temperature_{fold_tag}.json")
                        with open(T_json, "w", encoding="utf-8") as f:
                            json.dump({"best_T": float(best_T_local)}, f)
                except Exception:
                    pass
        except Exception as _e:
            print(f"[SAVE] threshold arrays failed: {_e}")
    except Exception as _e:
        print(f"[SAVE] Failed writing OUTPUT/result/metrics: {_e}")

    return auroc, auprc, precision, recall, f1, loss, (int(tn), int(fp), int(fn), int(tp))