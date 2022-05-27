from typing import List, Tuple, Dict
import json
import collections
import random


DataItem = collections.namedtuple("DataItem", ["seq_in", "seq_out","bio_seq_out" ,"domain"])

class RawDataLoaderBase:
    def __init__(self, *args, **kwargs):
        pass

    def load_data(self, path: str):
        pass

class FewShotExample(object):
    def __init__(
            self,
            gid: int,
            batch_id: int,
            domain_name: str,
            support_data_items: List[DataItem],
            test_data_items: List[DataItem],
    ):
        self.gid = gid
        self.batch_id = batch_id
        self.domain_name = domain_name

        self.support_data_items = support_data_items
        self.test_data_items = test_data_items

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'gid:{}\n\tdomain:{}\n\ttest_data:{}\n\tsupport_data:{}'.format(
            self.gid,
            self.domain_name,
            self.test_data_items,
            self.support_data_items,
        )

class FewShotRawDataLoader(RawDataLoaderBase):

    def load_data(self, path: str) -> (List[FewShotExample], List[List[FewShotExample]], int):
        """
            load few shot data set
            input:
                path: file path
            output
                examples: a list, all example loaded from path
                few_shot_batches: a list, of fewshot batch, each batch is a list of examples
                max_len: max sentence length
            """
        with open(path, 'r') as reader:
            raw_data = json.load(reader)
            #print(raw_data)
            examples, few_shot_batches, max_support_size = self.raw_data2examples(raw_data)
        return examples, few_shot_batches, max_support_size

    def raw_data2examples(self, raw_data: Dict) -> (List[FewShotExample], List[List[FewShotExample]], int):
        """
        process raw_data into examples
        """
        examples = []
        all_support_size = []
        few_shot_batches = []
        self.domain2id = {}
        for domain_n, domain in raw_data.items():
            self.domain2id[domain_n] = len(self.domain2id)
            # Notice: the batch here means few shot batch, not training batch
            for batch_id, batch in enumerate(domain):
                #print(batch_id)
                one_batch_examples = []
                support_data_items, test_data_items = self.batch2data_items(batch,domain_n)
                all_support_size.append(len(support_data_items))
                gid = len(examples)
                example = FewShotExample(
                    gid=gid,
                    batch_id=batch_id,
                    domain_name=domain_n,
                    support_data_items=support_data_items,
                    test_data_items=test_data_items,
                )
                examples.append(example)
                one_batch_examples.append(example)
                few_shot_batches.append(one_batch_examples)
        max_support_size = max(all_support_size)
        return examples, few_shot_batches, max_support_size

    def batch2data_items(self, batch: dict,domain_n) -> (List[DataItem], List[DataItem]):
        support_data_items = self.get_data_items(parts=batch['support'],domain_n=domain_n)
        test_data_items = self.get_data_items(parts=batch['query'],domain_n=domain_n)
        return support_data_items, test_data_items

    def get_data_items(self, parts: dict, domain_n) -> List[DataItem]:
        data_item_lst = []
        for seq_in, seq_out in zip(parts['seq_ins'], parts['seq_outs']):
            bio_seq_out = seq_out
            seq_out = [so.replace('B-','').replace('I-','') for so in seq_out]
            data_item = DataItem(seq_in=seq_in, seq_out=seq_out,bio_seq_out=bio_seq_out,domain=domain_n)
            data_item_lst.append(data_item)
            #print(data_item_lst)
        return data_item_lst
        