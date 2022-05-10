import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """ A custom dataset for a dataframe """

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long)
        }

class CustomDatasetTextPairs(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.term1 = dataframe.Term1
        self.term2 = dataframe.Term2
        self.target = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.term1)

    def __getitem__(self, index):
        term1 = str(self.term1[index])
        term2 = str(self.term2[index])

        inputs = self.tokenizer.encode_plus(
            term1,
            term2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.target[index], dtype=torch.float)
        }