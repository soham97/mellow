import pysepm
import os
import time
import json
from metrics import mmau
from metrics import aqa

class Metric:
    def __init__(self, task, sample_rate):
        self.task = task.lower()
        if 'mmau-test-mini' in self.task:
            jsonname = task.split(os.path.sep)[-1]
            with open(os.path.join("metrics","metadata",jsonname), 'r') as fp:
                self.metadata = json.load(fp)
    
    def compute_metrics(self, target_dict, output_dict, batch_length):
        src = target_dict["segment"].cpu().data.numpy()
        pred = output_dict["segment"]

        src = [x[:l] for x,l in zip(src,batch_length)]
        pred = [x[:l] for x,l in zip(pred,batch_length)]
        d = self.get_metrics(src,pred)

        for key in d.keys():
            self.metrics[key] += d[key]

    def get_metrics(self, generations, answers, filepaths):
        if 'mmau-test-mini' in self.task:
            self.metrics = mmau.evaluate_metric(generations, answers, self.metadata)
        elif 'mmau-test' in self.task:
            self.metrics = mmau.save_json(generations, answers, self.metadata)
        elif 'binary' in self.task:
            self.metrics = aqa.evaluate_metric_binary(generations, answers)
        elif 'aqa' in self.task:
            self.metrics = aqa.evaluate_metric(generations, answers)
        elif 'entail' in self.task:
            self.metrics = aqa.evaluate_entail_metric(generations, answers)
        elif 'caption_first' in self.task:
            from metrics import capmetrics
            fnames = [[f"fname_{i}"]*5 for i in range(int(len(generations)/5))]
            fnames = sum(fnames, [])
            captions_pred, captions_gt = capmetrics.parse_output_for_capmetrics(generations, answers, filepaths)
            for j in range(len(captions_pred)):
                captions_pred[j]["caption_predicted"] = captions_pred[j]["caption_predicted"].replace("The audio 1 is ","")
            # print(captions_pred)
            # print(captions_gt)
            metrics = capmetrics.evaluate_metrics(captions_pred, captions_gt, nb_reference_captions=5)
            metrics["main"] = metrics["spider"]
            self.metrics = metrics
        else:
            from metrics import capmetrics
            captions_pred, captions_gt = capmetrics.parse_output_for_diffmetrics(generations, answers)
            metrics = capmetrics.evaluate_metrics(captions_pred, captions_gt, nb_reference_captions=1)
            metrics["main"] = metrics["spider"]
            self.metrics = metrics
