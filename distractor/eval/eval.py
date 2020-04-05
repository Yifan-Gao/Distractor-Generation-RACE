from .bleu.bleu import Bleu
from collections import defaultdict
from argparse import ArgumentParser


class Eval:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self):
        print("="*5 , "MSCOCO Evaluation Script: ", "="*5)
        output = []
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            # (Rouge(), "ROUGE_L"),
            # (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    # print("%s: %0.2f"%(m, sc*100))
                    output.append(sc)
            else:
                # print("%s: %0.2f"%(method, score*100))
                output.append(score)
        return output

def eval(hyp, ref):
    """
        Given a filename, calculate the metric scores for that prediction file

        isDin: boolean value to check whether input file is DirectIn.txt
    """
    res1 = defaultdict(lambda: [])
    res2 = defaultdict(lambda: [])
    res3 = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for key, preds in hyp.items():
        res1[key] = [" ".join(preds[0])]
        res2[key] = [" ".join(preds[1])]
        res3[key] = [" ".join(preds[2])]
    for key, golds in ref.items():
        if key in res1.keys():
            for gold in golds:
                gts[key].append(" ".join(gold))

    print("*=" * 10, "First Distractor", "*=" * 10)
    DGEval1 = Eval(gts, res1)
    eval1 = list(map(lambda x: x * 100, DGEval1.evaluate()))
    print("B1: {:.2f}, B2: {:.2f}, B3: {:.2f}, B4: {:.2f}".format(
        eval1[0], eval1[1], eval1[2], eval1[3]
    ))


    print("*=" * 10, "Second Distractor", "*=" * 10)
    DGEval2 = Eval(gts, res2)
    eval2 = list(map(lambda x: x * 100, DGEval2.evaluate()))
    print("B1: {:.2f}, B2: {:.2f}, B3: {:.2f}, B4: {:.2f}".format(
        eval2[0], eval2[1], eval2[2], eval2[3]
    ))


    print("*=" * 10, "Third Distractor", "*=" * 10)
    DGEval3 = Eval(gts, res3)
    eval3 = list(map(lambda x: x * 100, DGEval3.evaluate()))
    print("B1: {:.2f}, B2: {:.2f}, B3: {:.2f}, B4: {:.2f}".format(
        eval3[0], eval3[1], eval3[2], eval3[3]
    ))
