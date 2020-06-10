import pickle
from typing import Dict

from pipeline_job import PipelineJob
from pytorch_pretrained_bert import BertTokenizer


class CreateIntegerizedCONLLTrainingData(PipelineJob):
    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                "data/benchmarks/aida-yago2-dataset/conll_dataset.pickle",
                f"data/versions/{opts.data_version_name}/indexes/popular_entity_to_id_dict.pickle",
            ],
            provides=[f"data/benchmarks/aida-yago2-dataset/conll_dataset_{opts.data_version_name}_{opts.create_integerized_training_instance_text_length}-{opts.create_integerized_training_instance_text_overlap}.pickle",],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        with open("data/benchmarks/aida-yago2-dataset/conll_dataset.pickle", "rb") as f:
            conll_dataset = pickle.load(f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/popular_entity_to_id_dict.pickle", "rb") as f:
            popular_entity_to_id_dict = pickle.load(f)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        # collect train valid test portionas and transform into our format

        train = list()
        valid = list()
        test = list()

        doc = None

        last_tok = None
        ents_in_doc = False
        split = None
        current_split_name = None
        bio_id = {'B': 0, 'I': 1, 'O': 2, }

        for item in conll_dataset:

            if current_split_name in ['train', 'valid', 'test'] and current_split_name != item['split'] and ents_in_doc:
                split.append(doc)
                doc = list()

            wiki_name = None
            bio = item['bio']

            if item['split'] == 'train':
                split = train
                current_split_name = 'train'
            if item['split'] == 'valid':
                split = valid
                current_split_name = 'valid'
            if item['split'] == 'test':
                split = test
                current_split_name = 'test'

            if item['sent_nr'] > 0 and item['tok_nr'] == 0 and last_tok != '.':
                doc.append(('.', tokenizer.convert_tokens_to_ids(['.'])[0], 'O', bio_id['O'], None, -1, len(doc)))

            if item['doc_start']:
                if doc is not None:
                    split.append(doc)
                doc = list()
                continue

            if item['wiki_name'] is not None and item['wiki_name'] != '--NME--':
                bio = 'O'
                for ent, c in item['wiki_name'].items():
                    # this statement only would not fire if add_missing_conll_entities was set to False
                    # during preprocessing
                    if ent in popular_entity_to_id_dict:
                        wiki_name = ent
                        bio = item['bio']
                        break

            # take care of all uppercase entity names, i.e. "LONDON"
            item_tok = item['tok']
            if item_tok.isupper() and item_tok not in tokenizer.vocab.keys():
                if (item_tok[0] + item_tok[1:].lower()) in tokenizer.vocab.keys():
                    item_tok = item_tok[0] + item_tok[1:].lower()
                elif item_tok.lower() in tokenizer.vocab.keys():
                    item_tok = item_tok.lower()

            last_bio = None
            for tok, tok_id in zip(tokenizer.tokenize(item_tok), tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item_tok))):
                wiki_id = popular_entity_to_id_dict[wiki_name] if wiki_name else -1
                if wiki_id >= 0:
                    if last_bio == 'B':
                        bio = 'I'
                else:
                    bio = 'O'
                doc.append((tok, tok_id, bio, bio_id[bio], wiki_name, wiki_id, len(doc)))
                last_bio = bio

            last_tok = item['tok']

        def create_overlapping_chunks(a_list, n, overlap):
            for i in range(0, len(a_list), n - overlap):
                yield a_list[i:i + n]

        train_overlapped = list()
        valid_overlapped = list()
        test_overlapped = list()

        for doc in train:
            train_overlapped.extend(
                create_overlapping_chunks(doc,
                                          self.opts.create_integerized_training_instance_text_length,
                                          self.opts.create_integerized_training_instance_text_overlap))

        for doc in valid:
            valid_overlapped.extend(
                create_overlapping_chunks(doc,
                                          self.opts.create_integerized_training_instance_text_length,
                                          self.opts.create_integerized_training_instance_text_overlap))

        for doc in test:
            test_overlapped.extend(
                create_overlapping_chunks(doc,
                                          self.opts.create_integerized_training_instance_text_length,
                                          self.opts.create_integerized_training_instance_text_overlap))

        with open(f"data/benchmarks/aida-yago2-dataset/conll_dataset_{self.opts.data_version_name}"
                  f"_{self.opts.create_integerized_training_instance_text_length}"
                  f"-{self.opts.create_integerized_training_instance_text_overlap}"
                  f".pickle", "wb") as f:
            pickle.dump((train_overlapped, valid_overlapped, test_overlapped), f)