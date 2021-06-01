#
# @author: Allan
#

from tqdm import tqdm
from common import Sentence, Instance
from typing import *
import re
import numpy as np
import json
from collections import defaultdict

from .transformers_converter import TransformersConverter
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Reader:
    def __init__(
        self,
        bert_model: str,
        O_tag: str = "O",
        kind: str = "entity",
        label_encoding: str = "BIOUL",
        drop_unannotated_sentences: bool = False,
        drop_unannotated_docs: bool = False,
        limit: int = None,
    ):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        self.digit2zero = False  # digit2zero -- we do not do this with bert
        self.vocab = set()
        self.subword_converter = TransformersConverter(bert_model) if bert_model else None
        self.maxlen = self.subword_converter.tokenizer.model_max_length if bert_model else None

        self.O = O_tag
        self.label_encoding = label_encoding
        self.kind = kind
        self.limit = limit
        self.assume_complete = True

        assert not (drop_unannotated_docs and drop_unannotated_sentences)
        self.drop_unannotated_sentences = drop_unannotated_sentences
        self.drop_unannotated_docs = drop_unannotated_docs

    def text_to_instances(self, tokens: List[str], annotations: List[Dict[str, Any]] = [], **metadata) -> Instance:
        metadata["og_tokens"] = tokens
        if self.subword_converter is not None:
            tokens, tokidx2bpeidxs = self.subword_converter(tokens)
        else:
            tokidx2bpeidxs = {i: [i] for i in range(len(tokens))}
        metadata["tokidx2bpeidxs"] = tokidx2bpeidxs
        tags = self.get_tags(tokens, annotations, metadata)

        for tokens, tags, metadata in self.as_maximal_subdocs(tokens, tags, metadata):
            inst = Instance(Sentence(tokens), tags)
            inst.metadata = metadata
            yield inst
            # tokens = [Token(t) for t in tokens]
            # tokens_field = TextField(tokens, self.token_indexers)
            # tag_namespace = "tags" if self.as_spans else "labels"
            # fields = dict(
            #     tokens=tokens_field,
            #     tags=SequenceLabelField(tags, tokens_field, label_namespace=tag_namespace),
            #     metadata=MetadataField(metadata),
            # )
            # if self.as_spans:
            #     spans = self.tags_as_spans(tags)
            #     fields["spans"] = ListField([LabelField(span) for span in spans])

            # yield Instance(fields)

    # def read_txt(self, file: str, number: int = -1) -> List[Instance]:
    #     print("Reading file: " + file)
    #     insts = []
    #     with open(file, "r", encoding="utf-8") as f:
    #         words = []
    #         labels = []
    #         for line in tqdm(f.readlines()):
    #             line = line.rstrip()
    #             if line == "":
    #                 inst = Instance(Sentence(words), labels)
    #                 inst.set_id(len(insts))
    #                 insts.append(inst)
    #                 words = []
    #                 labels = []
    #                 if len(insts) == number:
    #                     break
    #                 continue
    #             word, label = line.split()
    #             if self.digit2zero:
    #                 word = re.sub("\d", "0", word)  # replace digit with 0.
    #             words.append(word)
    #             self.vocab.add(word)
    #             labels.append(label)
    #     print("number of sentences: {}".format(len(insts)))
    #     return insts

    def read_txt(self, file_path: str, limit=-1) -> List[Instance]:
        og_n, actual_n = 0, 0
        insts = []
        with open(file_path) as f:
            for i, line in enumerate(f):
                if limit > 0 and i >= limit:
                    return
                datum = json.loads(line)
                tokens = datum.pop("tokens")
                annotations = datum.pop("gold_annotations")

                if self.drop_unannotated_sentences:
                    n = len(tokens)
                    og_n += n
                    tokens = self._get_annotated_subdoc(tokens, annotations, datum)
                    actual_n += len(tokens)
                    print(f"{i} Dropped from {n} tokens to {len(tokens)} annotated ones", flush=True)
                elif self.drop_unannotated_docs:
                    og_n += 1
                    if not annotations:
                        tokens = []
                        print(f"{i} Dropped unannotated doc", flush=True)
                    else:
                        actual_n += 1

                if tokens:
                    for instance in self.text_to_instances(tokens=tokens, annotations=annotations, **datum):
                        instance.set_id(len(insts))
                        insts.append(instance)

        if self.drop_unannotated_sentences:
            print(f"Cut down total tokens from {og_n} to {actual_n} = {100.*actual_n/og_n} %", flush=True)
        elif self.drop_unannotated_docs:
            print(f"Cut down docs from {og_n} to {actual_n} = {100.*actual_n/og_n} %", flush=True)

        return insts

    def get_tags(
        self,
        tokens: List[str],
        annotations: Dict[str, Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> List[str]:
        """Create tag sequence from annotations and possible subword mapping."""
        # Filter down annotations only to the specified kind
        annotations = [a for a in annotations if a["kind"] == self.kind]

        tokidx2bpeidxs = metadata["tokidx2bpeidxs"]
        n = len(tokens)
        # Default tags are either O or latent
        if self.assume_complete:
            tags = [self.O] * n
        else:
            tags = [self.latent_tag] * n
        # Fill in any partial annotations
        # Map start,ends for the observed annotations onto tokens
        # print("---")
        # print(metadata)
        # print(n, tokens)
        # print(annotations)
        # print(tokidx2bpeidxs)
        for ann in annotations:
            s, e, t = (
                tokidx2bpeidxs[ann["start"]][0],
                tokidx2bpeidxs[ann["end"] - 1][-1],
                ann["type"],
            )
            # print(s, e, t, ann)
            if t == self.O:
                for k in range(s, e + 1):
                    tags[k] = self.O
            else:
                if self.label_encoding == "BIOUL":
                    if e - s > 0:
                        tags[s] = f"B-{t}"
                        for k in range(s + 1, e):
                            tags[k] = f"I-{t}"
                        tags[e] = f"L-{t}"
                    else:
                        tags[s] = f"U-{t}"
                elif self.label_encoding == "BIO":
                    tags[s] = f"B-{t}"
                    if e - s > 0:
                        for k in range(s + 1, e + 1):
                            tags[k] = f"I-{t}"
                else:
                    raise ValueError(self.label_encoding)

        # If we are using transfomers, force first and last tokens (specials tokens) to be O tag
        if self.subword_converter is not None:
            tags[0] = self.O
            tags[-1] = self.O

        # print(tags, flush=True)

        return tags

    def tags_as_spans(self, tags: List[str]) -> List[str]:
        """Decode tagging into list of labels for span pairs.

        This is a simple implementation that assumes grammatical tagging.
        """
        # Decode back into annotated spans, but aligned to tokenization
        start, start_label = None, None
        spans_dict = dict()
        logger.debug(f"Tags are: {tags}")
        for k, tag in enumerate(tags):
            if tag in (self.O, self.latent_tag):
                if start is not None:
                    spans_dict[(start, k - 1)] = start_label
                    start, start_label = None, None
                elif tag == self.O:
                    spans_dict[(k, k)] = self.O
            else:
                marker, _, label = tag.partition("-")
                if marker == "U":
                    spans_dict[(k, k)] = label
                elif start is None:
                    start, start_label = k, label

        # Convert these to a list of all possible spans
        spans = []
        default_label = self.O if self.assume_complete else self.latent_tag
        n = len(tags)
        for i in range(n):
            for j in range(i, n):
                label = spans_dict.get((i, j), default_label)
                # if label not in (self.O, self.latent_tag):
                #     print(f"Got span {i,j,label}", flush=True)
                spans.append(label)

        return spans

    def as_maximal_subdocs(
        self,
        tokens: List[str],
        tags: List[str],
        metadata: Dict[str, Any],
    ) -> List[Tuple[List[str], List[str], Dict[str, Any]]]:
        """Break up docuemnts that are too large along sentence boundaries into ones that fit within the length limit."""
        if self.maxlen is None or len(tokens) <= self.maxlen:
            subdocs = [(tokens, tags, metadata)]
        else:
            subdocs = []
            tok2bpes = metadata.pop("tokidx2bpeidxs")
            bpe2tok = {v: k for k, vs in tok2bpes.items() for v in vs}
            uid = metadata.pop("uid", metadata.get("id", "NO ID"))
            # print(f"Breaking up sentences for {uid} with len: {len(tokens)}")
            s_tok, s_bpe, s_L = 0, 0, 0
            if "sentence_ends" in metadata:
                ends = metadata.pop("sentence_ends")
            else:
                print(f"No sentence ends found for {uid}, using crude length-based boundaries")
                ends = list(range(1, len(tok2bpes) + 1))
            subdoc_tokens, subdoc_tags, subdoc_metadata = None, None, None

            # We use a slightly smaller thatnactually ok subgrouping length because sometimes breaking up the sentences
            # that are too long starts or ends in the middle of a word and fixing to the full word goes over the maxlen
            maxlen = self.maxlen - 10
            for i, e_tok in enumerate(ends):
                # print(f"i:{i}, e:{e_tok}, bpes:{tok2bpes.keys()}")
                e_bpe = tok2bpes[e_tok][0] if e_tok < len(tok2bpes) else len(tokens)

                # Check to see if this sentence would put the current subdoc over the edge.
                if i and (e_bpe - s_bpe) > maxlen:
                    # If so, finish off the subdoc and advance the start cursors
                    subdoc_tok2bpe = {(k - s_tok): [v - s_bpe for v in tok2bpes[k]] for k in range(s_tok, e_tok)}
                    subdoc_metadata = dict(
                        uid=f"{uid}-S{s_L}:{i-1}",
                        tokidx2bpeidxs=subdoc_tok2bpe,
                        **metadata,
                    )
                    logger.debug(f"\nAdding subdoc {subdoc_metadata['uid']} with len {len(subdoc_tokens)}")
                    logger.debug(f'{" ".join([f"{t}/{l}" for t, l in zip(subdoc_tokens, subdoc_tags)])}')
                    assert len(subdoc_tokens) == len(subdoc_tags)
                    subdocs.append((subdoc_tokens, subdoc_tags, subdoc_metadata))
                    s_tok = ends[i - 1]
                    s_bpe = tok2bpes[s_tok][0]
                    s_L = i

                # Compute the next candidate subdoc
                # If the the next candidate subdoc will be too long on its own, break it up into smaller pieces that fit.
                # (ie, when there is sentence that is too long)
                if (e_bpe - s_bpe) > maxlen:
                    # Make sure the subgroups start/end on word boundaries
                    def to_word_boundary(bpe):
                        if bpe in (0, len(tokens)):
                            return bpe
                        else:
                            return tok2bpes[bpe2tok[bpe]][0]

                    n_groups = int(np.ceil((e_bpe - s_bpe) / maxlen))
                    s_bpes = [to_word_boundary(s_bpe + g * maxlen) for g in range(n_groups)]
                    e_bpes = [to_word_boundary(min(e_bpe, s_bpe + (g + 1) * maxlen)) for g in range(n_groups)]
                else:
                    s_bpes = [s_bpe]
                    e_bpes = [e_bpe]
                for g, (s_bpe, e_bpe) in enumerate(zip(s_bpes, e_bpes)):
                    # Collect the sentence tokens, tags, and add on start/end tokens&tags where needed
                    subdoc_tokens = tokens[s_bpe:e_bpe]
                    subdoc_tags = tags[s_bpe:e_bpe]
                    if not subdoc_tokens[0] == tokens[0]:
                        subdoc_tokens = [tokens[0]] + subdoc_tokens
                        subdoc_tags = [tags[0]] + subdoc_tags
                    if not subdoc_tokens[-1] == tokens[-1]:
                        subdoc_tokens = subdoc_tokens + [tokens[-1]]
                        subdoc_tags = subdoc_tags + [tags[-1]]

                    if g < len(s_bpes) - 1:
                        # All but the last group get turned into subdocs here
                        e_tok = bpe2tok[e_bpe]
                        subdoc_tok2bpe = {(k - s_tok): [v - s_bpe for v in tok2bpes[k]] for k in range(s_tok, e_tok)}
                        subdoc_metadata = dict(
                            uid=f"{uid}-S{s_L}:{i-1}.G{g}",
                            tokidx2bpeidxs=subdoc_tok2bpe,
                            **metadata,
                        )
                        logger.debug(f"\nAdding subdoc {subdoc_metadata['uid']} with len {len(subdoc_tokens)}")
                        logger.debug(f'{" ".join([f"{t}/{l}"  for t, l in zip(subdoc_tokens, subdoc_tags)])}')
                        assert len(subdoc_tokens) == len(subdoc_tags)
                        subdocs.append((subdoc_tokens, subdoc_tags, subdoc_metadata))
                    s_tok = e_tok

            # Add the last one
            subdoc_tok2bpe = {(k - s_tok): [v - s_bpe for v in tok2bpes[k]] for k in range(s_tok, e_tok)}
            subdoc_metadata = dict(uid=f"{uid}-S{s_L}:{i}", tokidx2bpeidxs=subdoc_tok2bpe, **metadata)
            # print(f"\nAdding subdoc {subdoc_metadata['uid']} with len {len(subdoc_tokens)}")
            # print(" ".join([f"{t}/{l}" if l != self.latent_tag else t for t, l in zip(subdoc_tokens, subdoc_tags)]))
            subdocs.append((subdoc_tokens, subdoc_tags, subdoc_metadata))

            cat_tokens = [t for (subdoctoks, _, _) in subdocs for t in subdoctoks[1:-1]]
            assert len(cat_tokens) == len(tokens) - 2, f"{len(cat_tokens)} != {len(tokens)-2}"
            assert cat_tokens == tokens[1:-1], f"{list(zip(cat_tokens, tokens[1:-1]))}"
        return subdocs

    def _get_annotated_subdoc(self, tokens, annotations, metadata):
        """ Chop off trailing sentences where there are no annotations. """
        annotations = [a for a in annotations if a["kind"] == self.kind]
        if annotations:
            last_ann = sorted(annotations, key=lambda a: -a["end"])[0]
            ends = sorted([e for e in metadata["sentence_ends"] if e >= last_ann["end"]])
            if ends:
                return tokens[: ends[0]]
            else:
                return tokens
        else:
            return []
