"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import json
from uuid import uuid4
import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)

try:
    import pointops
except:
    pointops = None


TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose and model is None:
            # if model is not none, trigger tester with trainer, no need to print config
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)





@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)




#### TEST
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
#####
# @TESTERS.register_module()
# class ClsVotingTester(TesterBase):
#     def __init__(
#         self,
#         num_repeat=1,
#         metric="allAcc",
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.num_repeat = 1
#         self.metric = metric
#         self.best_idx = 0
#         self.best_record = None
#         self.best_metric = 0

#     def test(self):
#         for i in range(self.num_repeat):
#             logger = get_root_logger()
#             logger.info(f">>>>>>>>>>>>>>>> Start Evaluation {i + 1} >>>>>>>>>>>>>>>>")
#             record = self.test_once()
#             if comm.is_main_process():
#                 if record[self.metric] > self.best_metric:
#                     self.best_record = record
#                     self.best_idx = i
#                     self.best_metric = record[self.metric]
#                 info = f"Current best record is Evaluation {i + 1}: "
#                 for m in self.best_record.keys():
#                     info += f"{m}: {self.best_record[m]:.4f} "
#                 logger.info(info)

#     def test_once(self):
#         logger = get_root_logger()
#         batch_time = AverageMeter()
#         intersection_meter = AverageMeter()
#         target_meter = AverageMeter()
#         record = {}
#         y_true = []
#         y_pred = []
#         self.model.eval()


#         all_categories=[]
#         all_targets_grouped=[]
#         all_predict_grouped = []    
#         current_row_predictions = [] 
#         current_row_targets=[]
#         count_cat=0

#         # alle test bilder einmal durchgehen
#         for idx, data_dict in enumerate(self.test_loader):
#             count_cat=count_cat+1
#             end = time.time()
#             data_dict = data_dict[0]  # current assume batch size is 1
#             voting_list = data_dict.pop("voting_list")
#             category = data_dict.pop("category")



            
#             #logger.info(f"all categories {all_categories[idx]}")
#             data_name = data_dict.pop("name")
          
#             input_dict = collate_fn(voting_list)
#             for key in input_dict.keys():
#                 if isinstance(input_dict[key], torch.Tensor):
#                     input_dict[key] = input_dict[key].cuda(non_blocking=True)
#             with torch.no_grad():
#                 pred = F.softmax(self.model(input_dict)["cls_logits"], -1).sum(
#                     0, keepdim=True
#                 )
               
#             pred = pred.max(1)[1].cpu().numpy()
#             y_pred.extend(pred.tolist())
#             y_true.extend([category.item()])
#             all_categories.append(category.item())


#             if all_categories[idx-1]!=all_categories[idx]: 
#                 logger.info(f"all categories momentan: {all_categories[idx]}")
#                 logger.info(f"anzahl pro cat: {count_cat}")

#                 all_targets_grouped.append(current_row_targets)
#                 logger.info(f"targets grouped momentaner Index: {all_targets_grouped}")
#                 current_row_targets=[]

#                 all_predict_grouped.append(current_row_predictions)
#                 logger.info(f"predictions grouped momentaner Index: {all_predict_grouped}")
#                 current_row_predictions = [] 
#                 count_cat=0
            
            
#             current_row_predictions.append(int(pred[0]))
#             current_row_targets.append(all_categories[idx])
#             #logger.info(f"current_row_predictions : {current_row_predictions}") 
#             #logger.info(f"current_row_targets : {current_row_targets}")

#             intersection, union, target = intersection_and_union(
#                 pred, category, self.cfg.data.num_classes, self.cfg.data.ignore_index
#             )


#             #logger.info(f"target : {target}")  #zb [0 0 0 1 0 ...]
#             #logger.info(f"target shape: {target.shape}")  #(40,)


#             intersection_meter.update(intersection)
#             target_meter.update(target)
#             record[data_name] = dict(intersection=intersection, target=target)
#             acc = sum(intersection) / (sum(target) + 1e-10)
#             m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
#             batch_time.update(time.time() - end)
#             logger.info(
#                 "Test: {} [{}/{}] "
#                 "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
#                 "Accuracy {acc:.4f} ({m_acc:.4f}) ".format(
#                     data_name,
#                     idx + 1,
#                     len(self.test_loader),
#                     batch_time=batch_time,
#                     acc=acc,
#                     m_acc=m_acc,
#                 )
#             )
     
#        # logger.info(f"iall_targets.shape: {all_targets.shape}")  # torch.Size([76022, 6])
#         # if current_row_predictions:
#         #     all_predict_grouped.append(current_row_predictions)
#         # if current_row_targets:
#         #     all_targets_grouped.append(current_row_targets)
#         all_predict_grouped = np.array(all_predict_grouped, dtype=object)
#         all_targets_grouped= np.array(all_targets_grouped, dtype=object)
#         logger.info(f"all_predict_grouped ersten zwei : {all_predict_grouped}")
#         logger.info(f"all_targets_grouped  ersten zwei : {all_targets_grouped}")
#         logger.info(f"all_predict_grouped am Ende: {all_predict_grouped.shape}")
#         logger.info(f"all_targets_grouped am Ende: {all_targets_grouped.shape}")
#         logger.info("Syncing ...")
#         comm.synchronize()
#         record_sync = comm.gather(record, dst=0)
#         # Listen von Listen zu flachen Listen machen
#         flat_preds = [item for sublist in all_predict_grouped for item in sublist]
#         flat_targets = [item for sublist in all_targets_grouped for item in sublist]

#         # Confusion Matrix berechnen
#         cm = confusion_matrix(flat_targets, flat_preds)




                
#         fig, ax = plt.subplots(figsize=(12, 12))  # größer machen
#         num_labels = cm.shape[0] if cm.shape[0] == cm.shape[1] else max(cm.shape)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_labels))
#         disp.plot(cmap='viridis', ax=ax, xticks_rotation=90)
#         plt.tight_layout()
#         plt.savefig("confusion_matrix_large.png", dpi=300)  # hohe Auflösung

#         if comm.is_main_process():
#             record = {}
#             for _ in range(len(record_sync)):
#                 r = record_sync.pop()
#                 record.update(r)
#                 del r
#             intersection = np.sum(
#                 [meters["intersection"] for _, meters in record.items()], axis=0
#             )
#             target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
#             accuracy_class = intersection / (target + 1e-10)
#             mAcc = np.mean(accuracy_class)
#             allAcc = sum(intersection) / (sum(target) + 1e-10)

#             logger.info("Val result: mAcc/allAcc {:.4f}/{:.4f}".format(mAcc, allAcc))
#             for i in range(self.cfg.data.num_classes):
#                 logger.info(
#                     "Class_{idx} - {name} Result: iou/accuracy {accuracy:.4f}".format(
#                         idx=i,
#                         name=self.cfg.data.names[i],
#                         accuracy=accuracy_class[i],
#                     )
#                 )
#             f1 = f1_score(y_true, y_pred, average="macro")  # oder "micro" oder "weighted" je nach Fokus
#             logger.info("F1-Score (macro): {:.4f}".format(f1))
#             return dict(mAcc=mAcc, allAcc=allAcc)

#     @staticmethod
#     def collate_fn(batch):
#         return batch
@TESTERS.register_module()
class ClsVotingTester(TesterBase):
    def __init__(self, num_repeat=1, metric="allAcc", sample_ratio=0.10, **kwargs):
        super().__init__(**kwargs)
        self.num_repeat = num_repeat
        self.metric = metric
        self.sample_ratio = float(sample_ratio)
        self.best_idx = 0
        self.best_record = None
        self.best_metric = -float("inf")

    def test(self):
        logger = get_root_logger()
        for i in range(self.num_repeat):
            logger.info(f">>>>>>>>>>>>>>>> Start Evaluation {i + 1} >>>>>>>>>>>>>>>>")
            record = self.test_once()
            if comm.is_main_process():
                score = record.get(self.metric, -float("inf"))
                if score > self.best_metric:
                    self.best_record = record
                    self.best_idx = i
                    self.best_metric = score

                info = f"Current best record is Evaluation {self.best_idx + 1}: "
                for k, v in self.best_record.items():
                    if isinstance(v, (float, int, np.floating, np.integer)):
                        info += f"{k}: {float(v):.4f} "
                logger.info(info)

    def test_once(self):
        print("BLUB LBIUB")
        logger = get_root_logger()
        batch_time = AverageMeter()

        num_classes = int(self.cfg.data.num_classes)
        ignore_index = int(self.cfg.data.ignore_index)
        eps = 1e-10

        # -----------------------------
        # 1) PRE-SCAN: pick 10% per class (by first occurrence in loader order)
        # -----------------------------
        per_class_indices = {c: [] for c in range(num_classes)}
        total_seen = 0

        for idx, data_dict in enumerate(self.test_loader):
            data_dict = data_dict[0]
            category = data_dict.get("category", None)
            if category is None:
                continue
            gt_class = int(category.item())
            if gt_class == ignore_index:
                continue
            if 0 <= gt_class < num_classes:
                per_class_indices[gt_class].append(idx)
                total_seen += 1

        keep_indices = set()
        keep_target_per_class = {}
        for c in range(num_classes):
            n = len(per_class_indices[c])
            k = int(np.ceil(self.sample_ratio * n))
            if n > 0:
                k = max(1, k)
            else:
                k = 0
            keep_target_per_class[c] = k
            keep_indices.update(per_class_indices[c][:k])

        logger.info(
            "Sampling for evaluation: ratio={:.2f}. Keeping {} / {} samples total.".format(
                self.sample_ratio, len(keep_indices), len(self.test_loader)
            )
        )
        for c in range(num_classes):
            name = self.cfg.data.names[c] if hasattr(self.cfg.data, "names") else str(c)
            logger.info(
                "Class_{:02d} - {}: keep {}/{} samples".format(
                    c, name, keep_target_per_class[c], len(per_class_indices[c])
                )
            )

        # -----------------------------
        # 2) EVALUATION PASS (only kept indices)
        # -----------------------------
        y_true_local = []
        y_pred_local = []
        first_sample_per_class_local = {}  # gt_class -> sample_dict (expects "coord")

        self.model.eval()

        for idx, data_dict in enumerate(self.test_loader):
            if idx not in keep_indices:
                continue

            end = time.time()
            data_dict = data_dict[0]  # assumes batch size is 1

            voting_list = data_dict.pop("voting_list")
            category = data_dict.pop("category")
            data_dict.pop("name", None)

            gt_class = int(category.item())
            if gt_class == ignore_index:
                continue

            if gt_class not in first_sample_per_class_local and len(voting_list) > 0:
                first_sample_per_class_local[gt_class] = voting_list[0]

            input_dict = collate_fn(voting_list)
            for key, val in input_dict.items():
                if isinstance(val, torch.Tensor):
                    input_dict[key] = val.cuda(non_blocking=True)

            with torch.no_grad():
                logits = self.model(input_dict)["cls_logits"]
                prob = F.softmax(logits, -1).sum(0, keepdim=True)  # aggregate votes
                pred = prob.max(1)[1]

            y_true_local.append(gt_class)
            y_pred_local.append(int(pred.item()))

            batch_time.update(time.time() - end)

        logger.info("Syncing ...")
        comm.synchronize()

        # Gather across ranks
        y_true_sync = comm.gather(y_true_local, dst=0)
        y_pred_sync = comm.gather(y_pred_local, dst=0)
        samples_sync = comm.gather(first_sample_per_class_local, dst=0)

        if not comm.is_main_process():
            return {}

        # Flatten
        y_true = [x for sub in y_true_sync for x in sub]
        y_pred = [x for sub in y_pred_sync for x in sub]

        # Merge first samples (keep first occurrence)
        first_sample_per_class = {}
        for d in samples_sync:
            for cls_id, sample in d.items():
                if cls_id not in first_sample_per_class:
                    first_sample_per_class[cls_id] = sample

        # -----------------------------
        # 3) CONFUSION MATRIX + METRICS
        # -----------------------------
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if t == ignore_index:
                continue
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1

        support = cm.sum(axis=1).astype(np.float64)
        tp = np.diag(cm).astype(np.float64)
        total = float(cm.sum() + eps)

        # Accuracy per class + overall
        acc_per_class = tp / (support + eps)
        allAcc = float(tp.sum() / total)

        # R1 (F1) per class + overall (macro + weighted)
        r1_per_class = np.zeros(num_classes, dtype=np.float64)
        for c in range(num_classes):
            tp_c = float(cm[c, c])
            fp_c = float(cm[:, c].sum() - cm[c, c])
            fn_c = float(cm[c, :].sum() - cm[c, c])

            precision = tp_c / (tp_c + fp_c + eps)
            recall = tp_c / (tp_c + fn_c + eps)
            r1_per_class[c] = 2.0 * precision * recall / (precision + recall + eps)

        r1_macro = float(np.mean(r1_per_class))
        r1_weighted = float((r1_per_class * support).sum() / (support.sum() + eps))

        # Always save confusion matrix (image + raw)
        self._save_confusion_matrix(cm)

        # Plot first kept test case per class (GT label)
        self._plot_first_samples_per_class(first_sample_per_class)

        # Logging summary
        logger.info("Val result: allAcc {:.4f}".format(allAcc))
        logger.info("Val result: R1_macro {:.4f} | R1_weighted {:.4f}".format(r1_macro, r1_weighted))

        for i in range(num_classes):
            name = self.cfg.data.names[i] if hasattr(self.cfg.data, "names") else str(i)
            logger.info(
                "Class_{idx} - {name} | Acc {acc:.4f} | R1 {r1:.4f} | Support {sup:.0f}".format(
                    idx=i,
                    name=name,
                    acc=float(acc_per_class[i]),
                    r1=float(r1_per_class[i]),
                    sup=float(support[i]),
                )
            )

        return dict(
            allAcc=allAcc,
            r1_macro=r1_macro,
            r1_weighted=r1_weighted,
        )

    def _save_confusion_matrix(self, cm: np.ndarray):
        """
        Saves:
          - confusion_matrix_large.png (always, using matplotlib only)
          - confusion_matrix.npy
          - confusion_matrix.csv
        """
        logger = get_root_logger()
        out_png = os.path.join(self.cfg.save_path, "confusion_matrix_large.png")
        out_npy = os.path.join(self.cfg.save_path, "confusion_matrix.npy")
        out_csv = os.path.join(self.cfg.save_path, "confusion_matrix.csv")

        try:
            np.save(out_npy, cm)
            np.savetxt(out_csv, cm, delimiter=",", fmt="%d")
        except Exception as e:
            logger.warning(f"Could not save confusion matrix array: {e}")

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 12))
            im = ax.imshow(cm, interpolation="nearest")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            n = cm.shape[0]
            ax.set_xticks(np.arange(n))
            ax.set_yticks(np.arange(n))

            # Optional names if available
            if hasattr(self.cfg.data, "names"):
                labels = list(self.cfg.data.names)
                ax.set_xticklabels(labels, rotation=90)
                ax.set_yticklabels(labels)
            else:
                ax.set_xticklabels([str(i) for i in range(n)], rotation=90)
                ax.set_yticklabels([str(i) for i in range(n)])

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(out_png, dpi=300)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not save confusion matrix plot: {e}")

    def _plot_first_samples_per_class(self, samples: dict):
        """
        Saves one plot per GT class: the first encountered *kept* test sample for that class.
        Expects sample dict contains:
          - "coord": (N, 3) tensor of xyz coordinates.
        """
        logger = get_root_logger()
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as e:
            logger.warning(f"Could not import matplotlib for class sample plots: {e}")
            return

        save_dir = os.path.join(self.cfg.save_path, "class_examples")
        os.makedirs(save_dir, exist_ok=True)

        for class_id in sorted(samples.keys()):
            sample = samples[class_id]

            if not isinstance(sample, dict) or "coord" not in sample:
                logger.warning(f"Sample for class {class_id} has no 'coord' key. Skipping plot.")
                continue

            coord = sample["coord"]
            if isinstance(coord, torch.Tensor):
                coord = coord.detach().cpu().numpy()

            if coord.ndim != 2 or coord.shape[1] < 3:
                logger.warning(f"Sample coord for class {class_id} has shape {coord.shape}. Skipping plot.")
                continue

            class_name = self.cfg.data.names[class_id] if hasattr(self.cfg.data, "names") else str(class_id)

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], s=1)
            ax.set_title(f"GT Class {class_id}: {class_name}")
            ax.set_axis_off()

            plt.tight_layout()
            safe_name = str(class_name).replace(" ", "_").replace("/", "_")
            out_path = os.path.join(save_dir, f"class_{class_id:02d}_{safe_name}.png")
            plt.savefig(out_path, dpi=300)
            plt.close(fig)

    @staticmethod
    def collate_fn(batch):
        return batch
