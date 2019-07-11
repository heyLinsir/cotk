import copy
import itertools
import random

import numpy as np
import pytest
import torch

from cotk.metric import MetricBase, \
	BleuPrecisionRecallMetric, EmbSimilarityPrecisionRecallMetric, \
	PerplexityMetric, MultiTurnPerplexityMetric, BleuCorpusMetric, SelfBleuCorpusMetric, \
	FwBwBleuCorpusMetric, MultiTurnBleuCorpusMetric, SingleTurnDialogRecorder, \
	MultiTurnDialogRecorder, LanguageGenerationRecorder, \
	MetricChain
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from metric_base import *

def setup_module():
	random.seed(0)
	np.random.seed(0)

def test_bleu_bug():
	ref = [[[1, 3], [3], [4]]]
	gen = [[1]]
	with pytest.raises(ZeroDivisionError):
		corpus_bleu(ref, gen, smoothing_function=SmoothingFunction().method7)

bleu_precision_recall_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_ref_len), "multi"),
	(zip(test_gen_len), "multi"),
	(zip(test_ngram), "add")
)

class TestBleuPrecisionRecallMetric():
	default_reference_key = 'candidate_allvocabs'
	default_gen_key = 'multiple_gen'
	default_keywords = (default_reference_key, default_gen_key)

	def test_base_class(self):
		with pytest.raises(NotImplementedError):
			dataloader = FakeMultiDataloader()
			gen = []
			reference = []
			bprm = BleuPrecisionRecallMetric(dataloader, 1, 3)
			super(BleuPrecisionRecallMetric, bprm)._score(gen, reference)

	def test_hashvalue(self):
		dataloader = FakeMultiDataloader()
		reference_key, gen_key = self.default_keywords
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=True, pad=False, \
								   ref_len='non-empty', gen_len='non-empty', test_prec_rec=True)
		bprm = BleuPrecisionRecallMetric(dataloader, 4, 3)
		assert bprm.candidate_allvocabs_key == reference_key
		bprm_shuffle = BleuPrecisionRecallMetric(dataloader, 4, 3)

		data_shuffle = shuffle_instances(data, self.default_keywords)
		for idx in range(len(data_shuffle[reference_key])):
			np.random.shuffle(data_shuffle[reference_key][idx])
		batches_shuffle = split_batch(data_shuffle, self.default_keywords)

		bprm.forward(data)
		res = bprm.close()

		for batch in batches_shuffle:
			bprm_shuffle.forward(batch)
		res_shuffle = bprm_shuffle.close()
		assert same_dict(res, res_shuffle, False)

		data_less_word = copy.deepcopy(data)
		data_less_word[reference_key][0][0] = data_less_word[reference_key][0][0][:-2]
		for data_unequal in [data_less_word] + generate_unequal_data(data, self.default_keywords, \
												  dataloader.pad_id, \
												  reference_key, reference_is_3D=True):
			bprm_unequal = BleuPrecisionRecallMetric(dataloader, 4, 3)

			bprm_unequal.forward(data_unequal)
			res_unequal = bprm_unequal.close()

			assert res['BLEU-4 hashvalue'] != res_unequal['BLEU-4 hashvalue']

	@pytest.mark.parametrize('argument, shape, type, batch_len, ref_len, gen_len, ngram', \
		bleu_precision_recall_test_parameter)
	def test_close(self, argument, shape, type, batch_len, ref_len, gen_len, ngram):
		dataloader = FakeMultiDataloader()

		if ngram not in range(1, 5):
			with pytest.raises(ValueError, match=r"ngram should belong to \[1, 4\]"):
				bprm = BleuPrecisionRecallMetric(dataloader, ngram, 3)
			return

		if argument == 'default':
			reference_key, gen_key = self.default_keywords
			bprm = BleuPrecisionRecallMetric(dataloader, ngram, 3)
		else:
			reference_key, gen_key = ('rk', 'gk')
			bprm = BleuPrecisionRecallMetric(dataloader, ngram, 3, reference_key, gen_key)

		# TODO: might need adaptation of dataloader.get_data for test_prec_rec
		# turn_length is not generated_num_per_context conceptually
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   ref_len=ref_len, gen_len=gen_len, test_prec_rec=True)
		_data = copy.deepcopy(data)
		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match="Batch num is not matched."):
				bprm.forward(data)
		else:
			bprm.forward(data)
			ans = bprm.close()
			prefix = 'BLEU-' + str(ngram)
			assert sorted(ans.keys()) == [prefix + ' hashvalue', prefix + ' precision', prefix + ' recall']

		assert same_dict(data, _data)



emb_similarity_precision_recall_test_parameter = generate_testcase( \
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_ref_len), "add"),
	(zip(test_gen_len), "add"),
	(zip(test_ref_vocab), "multi"),
	(zip(test_gen_vocab), "multi"),
	(zip(test_emb_mode), "add"),
	(zip(test_emb_type), "add"),
	(zip(test_emb_len), "add")
)


class TestEmbSimilarityPrecisionRecallMetric():
	default_reference_key = 'candidate_allvocabs'
	default_gen_key = 'multiple_gen'
	default_keywords = (default_reference_key, default_gen_key)

	def test_hashvalue(self):
		dataloader = FakeMultiDataloader()
		emb = {}
		emb_unequal = {}
		for word in dataloader.all_vocab_list[:dataloader.valid_vocab_len]:
			vec = []
			for j in range(5):
				vec.append(random.random())
			vec = np.array(vec)
			emb[word] = vec
			emb_unequal[word] = vec + 1

		reference_key, gen_key = self.default_keywords
		key_list = [reference_key, gen_key]
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=True, pad=False, \
								   ref_len='non-empty', gen_len='non-empty', \
								   ref_vocab='valid_vocab', gen_vocab='valid_vocab', test_prec_rec=True)
		espr = EmbSimilarityPrecisionRecallMetric(dataloader, emb, 'avg', 3)
		espr_shuffle = EmbSimilarityPrecisionRecallMetric(dataloader, emb, 'avg', 3)

		data_shuffle = shuffle_instances(data, key_list)
		for idx in range(len(data_shuffle[reference_key])):
			np.random.shuffle(data_shuffle[reference_key][idx])
		batches_shuffle = split_batch(data_shuffle, key_list)

		espr.forward(data)
		res = espr.close()

		for batch in batches_shuffle:
			espr_shuffle.forward(batch)
		res_shuffle = espr_shuffle.close()

		assert same_dict(res, res_shuffle, False)

		data_less_word = copy.deepcopy(data)
		data_less_word[reference_key][0][0] = data_less_word[reference_key][0][0][:-2]
		for data_unequal in [data_less_word] + generate_unequal_data(data, key_list, \
												dataloader.pad_id, \
												reference_key, reference_is_3D=True):
			espr_unequal = EmbSimilarityPrecisionRecallMetric(dataloader, emb, 'avg', 3)

			espr_unequal.forward(data_unequal)
			res_unequal = espr_unequal.close()

			assert res['avg-bow hashvalue'] != res_unequal['avg-bow hashvalue']
		espr_unequal = EmbSimilarityPrecisionRecallMetric(dataloader, emb_unequal, 'avg', 3)
		espr_unequal.forward(data)
		res_unequal = espr_unequal.close()
		assert res['avg-bow hashvalue'] != res_unequal['avg-bow hashvalue']

	@pytest.mark.parametrize('argument, shape, type, batch_len, ref_len, gen_len, ' \
							 'ref_vocab, gen_vocab, emb_mode, emb_type, emb_len', \
							 emb_similarity_precision_recall_test_parameter)
	def test_close(self, argument, shape, type, batch_len, ref_len, gen_len, \
							 ref_vocab, gen_vocab, emb_mode, emb_type, emb_len):
		dataloader = FakeMultiDataloader()

		emb = {}
		for word in dataloader.all_vocab_list[:dataloader.valid_vocab_len]:
			vec = []
			for j in range(5):
				vec.append(random.random())
			emb[word] = vec
		if emb_len == 'unequal':
			key = list(emb.keys())[0]
			emb[key] = emb[key][:-1]
		if emb_type == 'list':
			emb = np.array(list(emb.values()), dtype=np.float32).tolist()

		if emb_type != 'dict':
			with pytest.raises(ValueError, match="invalid type"):
				espr = EmbSimilarityPrecisionRecallMetric(dataloader, emb, emb_mode, 3)
			return
		else:
			if emb_len == 'unequal':
				with pytest.raises(ValueError, match="word embeddings have inconsistent embedding size or are empty"):
					espr = EmbSimilarityPrecisionRecallMetric(dataloader, emb, emb_mode, 3)
				return
		if emb_mode not in ['avg', 'extrema']:
			with pytest.raises(ValueError, match="mode should be 'avg' or 'extrema'."):
				espr = EmbSimilarityPrecisionRecallMetric(dataloader, emb, emb_mode, 3)
			return

		if argument == 'default':
			reference_key, gen_key = self.default_keywords
			print(emb)
			espr = EmbSimilarityPrecisionRecallMetric(dataloader, emb, emb_mode, 3)
		else:
			reference_key, gen_key = ('rk', 'gk')
			espr = EmbSimilarityPrecisionRecallMetric(dataloader, emb, emb_mode, 3, \
													  reference_key, gen_key)

		# TODO: might need adaptation of dataloader.get_data for test_prec_rec
		# turn_length is not generated_num_per_context conceptually
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   ref_len=ref_len, gen_len=gen_len, \
								   ref_vocab=ref_vocab, gen_vocab=gen_vocab, test_prec_rec=True)

		_data = copy.deepcopy(data)
		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match="Batch num is not matched."):
				espr.forward(data)
		else:
			# if emb_len < dataloader.all_vocab_size and \
			# 	(ref_vocab == 'all_vocab' or gen_vocab == 'all_vocab'):
			# 	with pytest.raises(ValueError, match="[a-z]* index out of range."):
			# 		espr.forward(data)
			# else:
			espr.forward(data)
			ans = espr.close()
			prefix = emb_mode + '-bow'
			assert sorted(ans.keys()) == [prefix + ' hashvalue', prefix + ' precision', prefix + ' recall']

		assert same_dict(data, _data)

perplexity_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_check), "add"),
	(zip(test_ref_len), "multi"),
	(zip(test_ref_vocab), "multi"),
	(zip(test_gen_prob_vocab), "multi"),
	(zip(test_resp_len), "add"),
	(zip(test_include_invalid), "multi")
)


perplexity_test_engine_parameter = generate_testcase(\
	(zip(test_ref_vocab), "multi"),
	(zip(test_gen_prob_vocab), "multi"),
)

class TestPerplexityMetric():
	default_reference_key = 'ref_allvocabs'
	default_reference_len_key = 'ref_length'
	default_gen_prob_key = 'gen_log_prob'
	default_keywords = (default_reference_key, default_reference_len_key, default_gen_prob_key)

	def get_perplexity(self, input, dataloader, invalid_vocab=False, \
						reference_key=default_reference_key, \
						reference_len_key=default_reference_len_key, \
						gen_prob_key=default_gen_prob_key):
		length_sum = 0
		word_loss = 0
		for i in range(len(input[reference_key])):
			max_length = input[reference_len_key][i]

			for j in range(max_length - 1):
				vocab_now = input[reference_key][i][j + 1]
				if vocab_now == dataloader.unk_id:
					continue
				if vocab_now < dataloader.vocab_size:
					word_loss += -(input[gen_prob_key][i][j][vocab_now])
				else:
					invalid_log_prob = input[gen_prob_key][i][j][dataloader.unk_id] - \
									 np.log(dataloader.all_vocab_size - dataloader.vocab_size)
					if invalid_vocab:
						word_loss += -np.log(np.exp(invalid_log_prob) + \
											np.exp(input[gen_prob_key][i][j][vocab_now]))
					else:
						word_loss += -invalid_log_prob
				length_sum += 1
		# print('test_metric.word_loss: ', word_loss)
		# print('test_metric.length_sum: ', 	length_sum)
		return np.exp(word_loss / length_sum)

	@pytest.mark.parametrize('to_list, pad', [[True, False], [True, True], [False, True]])
	def test_hashvalue(self, to_list, pad):
		dataloader = FakeDataLoader()
		reference_key, reference_len_key, gen_prob_key = self.default_keywords
		key_list = [reference_key, reference_len_key, gen_prob_key]
		data = dataloader.get_data(reference_key=reference_key, \
								   reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=to_list, pad=pad, \
								   gen_prob_check='no_check', ref_len='non-empty', \
								   ref_vocab='non-empty', gen_prob_vocab='all_vocab', \
								   resp_len='>=2')
		pm = PerplexityMetric(dataloader, invalid_vocab=True, full_check=False)
		pm_shuffle = PerplexityMetric(dataloader, invalid_vocab=True, full_check=False)

		data_shuffle = shuffle_instances(data, key_list)

		batches_shuffle = split_batch(data_shuffle, key_list, \
									  to_list=to_list, less_pad=pad, \
									  reference_key=reference_key, reference_is_3D=False)

		pm.forward(data)
		res = pm.close()

		for batch in batches_shuffle:
			pm_shuffle.forward(batch)
		res_shuffle = pm_shuffle.close()

		assert same_dict(res, res_shuffle, False)

		for data_unequal in generate_unequal_data(data, key_list, dataloader.pad_id, \
												  reference_key, reference_len_key, \
												  reference_is_3D=False):
			pm_unequal = PerplexityMetric(dataloader, invalid_vocab=True, full_check=False)

			pm_unequal.forward(data_unequal)
			res_unequal = pm_unequal.close()

			assert res['perplexity hashvalue'] != res_unequal['perplexity hashvalue']

	@pytest.mark.parametrize("ref_vocab, gen_prob_vocab", perplexity_test_engine_parameter)
	def test_same_result_with_pytorch_engine(self, ref_vocab, gen_prob_vocab):
		dataloader = FakeDataLoader()
		reference_key, reference_len_key, gen_prob_key = self.default_keywords
		data = dataloader.get_data(reference_key=reference_key, \
								   reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=True, pad=True, \
								   gen_prob_check='no_check', ref_len='non-empty', \
								   ref_vocab=ref_vocab, gen_prob_vocab=gen_prob_vocab, \
								   resp_len='>=2')
		pm = PerplexityMetric(dataloader, invalid_vocab=gen_prob_vocab == "all_vocab", full_check=False)
		pm_shuffle = PerplexityMetric(dataloader, invalid_vocab=gen_prob_vocab == "all_vocab", full_check=False)
		pm_shuffle2 = PerplexityMetric(dataloader, invalid_vocab=gen_prob_vocab == "all_vocab", full_check=False)

		data_shuffle = copy.deepcopy(data)
		indices = list(range(len(data_shuffle[reference_key])))
		np.random.shuffle(indices)
		data_shuffle[reference_len_key] = list(np.array(data_shuffle[reference_len_key])[indices])
		data_shuffle[gen_prob_key] = torch.Tensor(np.array(data_shuffle[gen_prob_key])[indices])

		pm.forward(data)
		res = pm.close()

		data_shuffle[reference_key] = np.array(data_shuffle[reference_key])[indices]
		pm_shuffle.forward(data_shuffle)
		res_shuffle = pm_shuffle.close()

		data_shuffle[reference_key] = torch.LongTensor(data_shuffle[reference_key])
		pm_shuffle2.forward(data_shuffle)
		res_shuffle2 = pm_shuffle2.close()

		assert res['perplexity hashvalue'] == res_shuffle['perplexity hashvalue']
		assert res['perplexity hashvalue'] == res_shuffle2['perplexity hashvalue']
		assert np.isclose(res['perplexity'], res_shuffle['perplexity'])
		assert np.isclose(res['perplexity'], res_shuffle2['perplexity'])

	@pytest.mark.parametrize( \
		'argument, shape, type, batch_len, check, ref_len, ref_vocab, gen_prob_vocab, resp_len, include_invalid', \
		perplexity_test_parameter)
	def test_close(self, argument, shape, type, batch_len, check, \
				   ref_len, ref_vocab, gen_prob_vocab, resp_len, include_invalid):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random_check' or 'full_check' or 'no_check'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeDataLoader()
		reference_key, reference_len_key, gen_prob_key = self.default_keywords \
			if argument == 'default' else ('ra', 'rl', 'glp')
		data = dataloader.get_data(reference_key=reference_key, \
								   reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_prob_check=check, ref_len=ref_len, \
								   ref_vocab=ref_vocab, gen_prob_vocab=gen_prob_vocab, \
								   resp_len=resp_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			pm = PerplexityMetric(dataloader, invalid_vocab=include_invalid, full_check=(check=='full_check'))
		else:
			pm = PerplexityMetric(dataloader, reference_key, reference_len_key, gen_prob_key, \
								   invalid_vocab=include_invalid,  full_check=(check=='full_check'))

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num'):
				pm.forward(data)
		elif check == 'no_check':
			if resp_len == '<2':
				with pytest.raises(ValueError, match='resp_length must no less than 2,' \
													 ' because <go> and <eos> are always included.'):
					pm.forward(data)
			elif include_invalid != (gen_prob_vocab == 'all_vocab'):
				with pytest.raises(ValueError):
					pm.forward(data)
			else:
				pm.forward(data)
				assert np.isclose(pm.close()['perplexity'], \
								  self.get_perplexity(data, dataloader, include_invalid, \
													  reference_key, reference_len_key, gen_prob_key))
		else:
			with pytest.raises(ValueError, \
							   match=r'data\[gen_log_prob_key\] must be processed after log_softmax.'):
				pm.forward(data)
		assert same_dict(data, _data)

multiperplexity_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_check), "add"),
	(zip(test_ref_len), "multi"),
	(zip(test_ref_vocab), "multi"),
	(zip(test_gen_prob_vocab), "multi"),
	(zip(test_resp_len), "multi"),
	(zip(test_include_invalid), "multi")
)


class TestMultiTurnPerplexityMetric:
	default_reference_key = 'multi_turn_ref_allvocabs'
	default_reference_len_key = 'multi_turn_ref_length'
	default_gen_prob_key = 'multi_turn_gen_log_prob'
	default_keywords = (default_reference_key, default_reference_len_key, default_gen_prob_key)

	def get_perplexity(self, input, dataloader, invalid_vocab=False, \
					   reference_key=default_reference_key, \
					   reference_len_key=default_reference_len_key, \
					   gen_prob_key=default_gen_prob_key):
		length_sum = 0
		word_loss = 0
		for i in range(len(input[reference_key])):
			for turn in range(len(input[reference_key][i])):
				max_length = input[reference_len_key][i][turn]
				gen_prob_turn = input[gen_prob_key][i][turn]
				for j in range(max_length - 1):
					vocab_now = input[reference_key][i][turn][j + 1]
					if vocab_now == dataloader.unk_id:
						continue
					if vocab_now < dataloader.vocab_size:
						word_loss += -(gen_prob_turn[j][vocab_now])
					else:
						invalid_log_prob = gen_prob_turn[j][dataloader.unk_id] - \
										 np.log(dataloader.all_vocab_size - dataloader.vocab_size)
						if invalid_vocab:
							word_loss += -np.log(np.exp(invalid_log_prob) + \
												np.exp(gen_prob_turn[j][vocab_now]))
						else:
							word_loss += -invalid_log_prob
					length_sum += 1
		return np.exp(word_loss / length_sum)

	@pytest.mark.parametrize('to_list, pad', [[True, False], [True, True], [False, True]])
	def test_hashvalue(self, to_list, pad):
		dataloader = FakeMultiDataloader()
		reference_key, reference_len_key, gen_prob_key = self.default_keywords
		key_list = [reference_key, reference_len_key, gen_prob_key]
		data = dataloader.get_data(reference_key=reference_key, \
								   reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=to_list, pad=pad, \
								   gen_prob_check='no_check', ref_len='non-empty', \
								   ref_vocab='non-empty', gen_prob_vocab='valid_vocab', \
								   resp_len=">=2")

		mtpm = MultiTurnPerplexityMetric(dataloader, invalid_vocab=False, full_check=False)
		mtpm_shuffle = MultiTurnPerplexityMetric(dataloader, invalid_vocab=False, full_check=False)

		data_shuffle = shuffle_instances(data, key_list)

		batches_shuffle = split_batch(data_shuffle, key_list, \
									  less_pad=pad, to_list=to_list, \
									  reference_key=reference_key, reference_is_3D=True)

		mtpm.forward(data)
		res = mtpm.close()

		for batch in batches_shuffle:
			mtpm_shuffle.forward(batch)
		res_shuffle = mtpm_shuffle.close()

		assert same_dict(res, res_shuffle, False)

		for data_unequal in generate_unequal_data(data, key_list, dataloader.pad_id, \
												  reference_key, reference_len_key, \
												  reference_is_3D=True):
			mtpm_unequal = MultiTurnPerplexityMetric(dataloader, invalid_vocab=False, full_check=False)

			mtpm_unequal.forward(data_unequal)
			res_unequal = mtpm_unequal.close()

			assert res['perplexity hashvalue'] != res_unequal['perplexity hashvalue']

	@pytest.mark.parametrize( \
		'argument, shape, type, batch_len, check, ref_len, ref_vocab, gen_prob_vocab, resp_len, include_invalid', \
		multiperplexity_test_parameter)
	def test_close(self, argument, shape, type, batch_len, check, \
				   ref_len, ref_vocab, gen_prob_vocab, resp_len, include_invalid):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random_check' or 'full_check' or 'no_check'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeMultiDataloader()
		reference_key, reference_len_key, gen_prob_key = self.default_keywords \
			if argument == 'default' else ('rk', 'rl', 'gp')
		data = dataloader.get_data(reference_key=reference_key, \
								   reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_prob_check=check, ref_len=ref_len, \
								   ref_vocab=ref_vocab, gen_prob_vocab=gen_prob_vocab, \
								   resp_len = resp_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtpm = MultiTurnPerplexityMetric(dataloader, \
											 invalid_vocab=include_invalid, full_check=(check=='full_check'))
		else:
			mtpm = MultiTurnPerplexityMetric(dataloader, reference_key, reference_len_key, gen_prob_key, \
								   invalid_vocab=include_invalid,  full_check=(check=='full_check'))

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtpm.forward(data)
		elif check == 'no_check':
			if resp_len == '<2':
				with pytest.raises(ValueError, match='resp_length must no less than 2,' \
													 ' because <go> and <eos> are always included.'):
					mtpm.forward(data)
			elif include_invalid != (gen_prob_vocab == 'all_vocab'):
				with pytest.raises(ValueError):
					mtpm.forward(data)
			else:
				mtpm.forward(data)
				assert np.isclose(mtpm.close()['perplexity'], \
								  self.get_perplexity(data, dataloader, include_invalid, \
													  reference_key, reference_len_key, gen_prob_key))
		else:
			with pytest.raises(ValueError, \
							   match=r'data\[gen_log_prob_key\] must be processed after log_softmax.'):
				mtpm.forward(data)
		assert same_dict(data, _data)


bleu_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)


class TestBleuCorpusMetric:
	default_reference_key = "ref_allvocabs"
	default_gen_key = "gen"
	default_keywords = [default_reference_key, default_gen_key]

	def get_bleu(self, dataloader, input, reference_key, gen_key):
		refs = []
		gens = []
		for gen_sen, resp_sen in zip(input[gen_key], input[reference_key]):
			gen_sen_processed = dataloader.trim_index(gen_sen)
			resp_sen_processed = dataloader.trim_index(resp_sen[1:])
			refs.append([resp_sen_processed])
			gens.append(gen_sen_processed)
		gens = replace_unk(gens)
		return corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7)

	@pytest.mark.parametrize('to_list, pad', [[True, False], [True, True], [False, True]])
	def test_hashvalue(self, to_list, pad):
		dataloader = FakeDataLoader()
		reference_key, gen_key = self.default_keywords
		key_list = [reference_key, gen_key]
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=to_list, pad=pad, \
								   gen_len='non-empty', ref_len='non-empty')
		bcm = BleuCorpusMetric(dataloader)
		bcm_shuffle = BleuCorpusMetric(dataloader)

		data_shuffle = shuffle_instances(data, key_list)
		batches_shuffle = split_batch(data_shuffle, key_list, \
									  less_pad=pad, to_list=to_list, \
									  reference_key=reference_key, reference_is_3D=False)

		bcm.forward(data)
		res = bcm.close()

		for batch in batches_shuffle:
			bcm_shuffle.forward(batch)
		res_shuffle = bcm_shuffle.close()

		assert same_dict(res, res_shuffle, False)

		for data_unequal in generate_unequal_data(data, key_list, dataloader.pad_id, \
												  reference_key, reference_is_3D=False):
			bcm_unequal = BleuCorpusMetric(dataloader)

			bcm_unequal.forward(data_unequal)
			res_unequal = bcm_unequal.close()

			assert res['bleu hashvalue'] != res_unequal['bleu hashvalue']

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len, ref_len', bleu_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeDataLoader()
		reference_key, gen_key = self.default_keywords \
			if argument == 'default' else ('rk', 'gk')
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			bcm = BleuCorpusMetric(dataloader)
		else:
			bcm = BleuCorpusMetric(dataloader, reference_allvocabs_key=reference_key, gen_key=gen_key)

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				bcm.forward(data)
		else:
				bcm.forward(data)
				assert np.isclose(bcm.close()['bleu'], self.get_bleu(dataloader, data, reference_key, gen_key))
		assert same_dict(data, _data)

	def test_bleu_bug(self):
		dataloader = FakeDataLoader()
		ref = [[2, 5, 3]]
		gen = [[5]]
		data = {self.default_reference_key: ref, self.default_gen_key: gen}
		bcm = BleuCorpusMetric(dataloader)

		with pytest.raises(ZeroDivisionError):
			bcm.forward(data)
			bcm.close()

self_bleu_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(["non-empty"]), "multi"),
)


class TestSelfBleuCorpusMetric:
	def get_self_bleu(self, dataloader, input, gen_key):
		gens = []
		for gen_sen in input[gen_key]:
			gen_sen_processed = dataloader.trim_index(gen_sen)
			gens.append(gen_sen_processed)
		refs = copy.deepcopy(gens)
		_refs = replace_unk(refs)
		bleu_irl = []
		for i in range(len(gens)):
			bleu_irl.append(sentence_bleu(
				refs[:i]+refs[i+1:], _refs[i], smoothing_function=SmoothingFunction().method1))
		return 1.0 * sum(bleu_irl) / len(bleu_irl)

	def test_hashvalue(self):
		dataloader = FakeDataLoader()
		gen_key = 'gen'
		key_list = [gen_key]
		data = dataloader.get_data(gen_key=gen_key, \
								   to_list=False, \
								   pad=True, \
								   gen_len='non-empty')
		bcm = SelfBleuCorpusMetric(dataloader)
		bcm_shuffle = SelfBleuCorpusMetric(dataloader)
		bcm_unequal = SelfBleuCorpusMetric(dataloader, sample=2)

		data_shuffle = shuffle_instances(data, key_list)
		batches_shuffle = split_batch(data_shuffle, key_list)

		bcm.forward(data)
		res = bcm.close()

		for batch in batches_shuffle:
			bcm_shuffle.forward(batch)
		res_shuffle = bcm_shuffle.close()

		assert same_dict(res, res_shuffle, exact_equal=False)

		bcm_unequal.forward(data)
		res_unequal = bcm_unequal.close()

		assert res['self-bleu hashvalue'] != res_unequal['self-bleu hashvalue']

	@pytest.mark.parametrize('argument, shape, type, gen_len', self_bleu_test_parameter)
	def test_close(self, argument, shape, type, gen_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeDataLoader()
		gen_key = 'gen' \
			if argument == 'default' else 'gk'
		data = dataloader.get_data(gen_key=gen_key, \
								   to_list=(type == 'list'), \
								   pad=(shape == 'pad'), \
								   gen_len=gen_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			bcm = SelfBleuCorpusMetric(dataloader)
		else:
			bcm = SelfBleuCorpusMetric(dataloader, gen_key)

		bcm.forward(data)
		assert np.isclose(bcm.close()['self-bleu'], self.get_self_bleu(dataloader, data, gen_key))
		assert same_dict(data, _data)

	# def test_self_bleu_bug(self):
	# 	dataloader = FakeDataLoader()
	# 	gen = [[1]]
	# 	data = {'gen': gen}
	# 	bcm = SelfBleuCorpusMetric(dataloader)

	# 	with pytest.raises(ZeroDivisionError):
	# 		bcm.forward(data)
	# 		bcm.close()

fwbw_bleu_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(["non-empty"]), "multi"),
	(zip(["non-empty"]), "multi")
)

class TestFwBwBleuCorpusMetric:
	def get_bleu(self, dataloader, input, reference_key, gen_key):
		refs = []
		gens = []
		for gen_sen, resp_sen in zip(input[gen_key], input[reference_key]):
			gen_sen_processed = dataloader.trim_index(gen_sen)
			resp_sen_processed = dataloader.trim_index(resp_sen[1:])
			refs.append(resp_sen_processed)
			gens.append(gen_sen_processed)
		gens = replace_unk(gens)
		bleu_irl_bw, bleu_irl_fw = [], []
		for i in range(len(gens)):
			bleu_irl_fw.append(sentence_bleu(refs, gens[i], smoothing_function=SmoothingFunction().method1))
		for i in range(len(refs)):
			bleu_irl_bw.append(sentence_bleu(gens, refs[i], smoothing_function=SmoothingFunction().method1))

		fw_bleu = (1.0 * sum(bleu_irl_fw) / len(bleu_irl_fw))
		bw_bleu = (1.0 * sum(bleu_irl_bw) / len(bleu_irl_bw))
		return 2.0 * bw_bleu * fw_bleu / (fw_bleu + bw_bleu)

	@pytest.mark.parametrize('to_list, pad', [[True, False], [True, True], [False, True]])
	def test_hashvalue(self, to_list, pad):
		dataloader = FakeDataLoader()
		reference_key, gen_key = ('resp_allvocabs', 'gen')
		key_list = [reference_key, gen_key]
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=to_list, pad=pad, \
								   gen_len='non-empty', ref_len='non-empty')

		dataloader.data["test"][reference_key] = data[reference_key]
		bcm = FwBwBleuCorpusMetric(dataloader, reference_key)
		bcm.forward(data)
		res = bcm.close()

		data_shuffle = shuffle_instances(data, key_list)
		dataloader.data["test"][reference_key] = data_shuffle[reference_key]
		bcm_shuffle = FwBwBleuCorpusMetric(dataloader, reference_key)
		bcm_shuffle.forward(data_shuffle)
		res_shuffle = bcm_shuffle.close()

		assert same_dict(res, res_shuffle, False)

		for data_unequal in generate_unequal_data(data, key_list, dataloader.pad_id, \
												  reference_key, reference_is_3D=False):
			dataloader.data["test"][reference_key] = data_unequal[reference_key]
			bcm_unequal = FwBwBleuCorpusMetric(dataloader, reference_key)

			bcm_unequal.forward(data_unequal)
			res_unequal = bcm_unequal.close()
			assert res['fw-bw-bleu hashvalue'] != res_unequal['fw-bw-bleu hashvalue']
		bcm_unequal = FwBwBleuCorpusMetric(dataloader, reference_key, sample=2)
		bcm_unequal.forward(data)
		res_unequal = bcm_unequal.close()
		assert res['fw-bw-bleu hashvalue'] != res_unequal['fw-bw-bleu hashvalue']

	@pytest.mark.parametrize('argument, shape, type, gen_len, ref_len', fwbw_bleu_test_parameter)
	def test_close(self, argument, shape, type, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeDataLoader()
		reference_key, gen_key = ('resp_allvocabs', 'gen') \
			if argument == 'default' else ('rk', 'gk')
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		dataloader.data["test"][reference_key] = data[reference_key]
		_data = copy.deepcopy(data)
		if argument == 'default':
			bcm = FwBwBleuCorpusMetric(dataloader, reference_key)
		else:
			bcm = FwBwBleuCorpusMetric(dataloader, reference_key, gen_key)

		bcm.forward(data)
		assert np.isclose(bcm.close()['fw-bw-bleu'], self.get_bleu(dataloader, data, reference_key, gen_key))
		assert same_dict(data, _data)

	# def test_fwbwbleu_bug(self):
	# 	dataloader = FakeDataLoader()
	# 	ref = [[2, 1, 3]]
	# 	gen = [[1]]
	# 	reference_key = 'resp_allvocabs'
	# 	data = {reference_key: ref, 'gen': gen}
	# 	dataloader.data["test"][reference_key] = data[reference_key]
	# 	bcm = FwBwBleuCorpusMetric(dataloader, reference_key)

	# 	with pytest.raises(ZeroDivisionError):
	# 		bcm.forward(data)
	# 		bcm.close()


multi_bleu_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)



class TestMultiTurnBleuCorpusMetric:
	default_reference_key = "reference_allvocabs"
	default_turn_len_key = "turn_length"
	default_gen_key = "multi_turn_gen"
	default_keywords = [default_reference_key, default_turn_len_key, default_gen_key]

	def get_bleu(self, dataloader, input, reference_key, gen_key):
		refs = []
		gens = []
		for i in range(len(input[reference_key])):
			for resp_sen, gen_sen in zip(input[reference_key][i], input[gen_key][i]):
				gen_sen_processed = dataloader.trim_index(gen_sen)
				resp_sen_processed = dataloader.trim_index(resp_sen)
				gens.append(gen_sen_processed)
				refs.append([resp_sen_processed[1:]])
		gens = replace_unk(gens)
		return corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7)

	@pytest.mark.parametrize('to_list, pad', [[True, False], [True, True], [False, True]])
	def test_hashvalue(self, to_list, pad):
		dataloader = FakeMultiDataloader()
		reference_key, turn_len_key, gen_key = self.default_keywords
		key_list = [reference_key, turn_len_key, gen_key]
		data = dataloader.get_data(reference_key=reference_key, turn_len_key=turn_len_key, gen_key=gen_key, \
								   to_list=to_list, pad=pad, ref_len='non-empty', \
								   ref_vocab='non-empty')

		mtbcm = MultiTurnBleuCorpusMetric(dataloader)
		mtbcm_shuffle = MultiTurnBleuCorpusMetric(dataloader)

		data_shuffle = shuffle_instances(data, key_list)
		batches_shuffle = split_batch(data_shuffle, key_list, \
									  less_pad=pad, to_list=to_list, \
									  reference_key=reference_key, reference_is_3D=True)

		mtbcm.forward(data)
		res = mtbcm.close()

		for batch in batches_shuffle:
			mtbcm_shuffle.forward(batch)
		res_shuffle = mtbcm_shuffle.close()
		assert same_dict(res, res_shuffle, False)

		data_less_word = copy.deepcopy(data)
		for idx, turn_len in enumerate(data_less_word[turn_len_key]):
			if turn_len > 1:
				data_less_word[turn_len_key][idx] -= 1
		for data_unequal in [data_less_word] + generate_unequal_data(data, key_list, dataloader.pad_id, \
												  reference_key=reference_key, reference_is_3D=True):
			mtbcm_unequal = MultiTurnBleuCorpusMetric(dataloader)

			mtbcm_unequal.forward(data_unequal)
			res_unequal = mtbcm_unequal.close()

			assert res['bleu hashvalue'] != res_unequal['bleu hashvalue']

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len, ref_len', multi_bleu_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeMultiDataloader()
		reference_key, turn_len_key, gen_key = self.default_keywords \
			if argument == 'default' else ('rk', 'tlk', 'gk')
		data = dataloader.get_data(reference_key=reference_key, turn_len_key=turn_len_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtbcm = MultiTurnBleuCorpusMetric(dataloader)
		else:
			mtbcm = MultiTurnBleuCorpusMetric(dataloader, multi_turn_reference_allvocabs_key=reference_key, \
											  multi_turn_gen_key=gen_key, turn_len_key=turn_len_key)

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtbcm.forward(data)
		else:
			mtbcm.forward(data)
			assert np.isclose(mtbcm.close()['bleu'], self.get_bleu(dataloader, data, reference_key, gen_key))
		assert same_dict(data, _data)

	def test_bleu(self):
		dataloader = FakeMultiDataloader()
		ref = [[[2, 5, 3]]]
		gen = [[[5]]]
		turn_len = [1]
		data = {self.default_reference_key: ref, self.default_gen_key: gen, self.default_turn_len_key: turn_len}
		mtbcm = MultiTurnBleuCorpusMetric(dataloader)

		with pytest.raises(ZeroDivisionError):
			mtbcm.forward(data)
			mtbcm.close()


single_turn_dialog_recorder_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)

class TestSingleTurnDialogRecorder():
	default_post_key = "post_allvocabs"
	default_ref_key = "resp_allvocabs"
	default_gen_key = "gen"
	default_keywords = [default_post_key, default_ref_key, default_gen_key]

	def get_sen_from_index(self, dataloader, data, post_key=default_post_key, \
			reference_key=default_ref_key, gen_key=default_gen_key):
		ans = { \
			'post': [], \
			'resp': [], \
			'gen': [], \
			}
		for sen in data[post_key]:
			ans['post'].append(dataloader.convert_ids_to_tokens(sen[1:]))
		for sen in data[reference_key]:
			ans['resp'].append(dataloader.convert_ids_to_tokens(sen[1:]))
		for sen in data[gen_key]:
			ans['gen'].append(dataloader.convert_ids_to_tokens(sen))

		return ans

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len, ref_len', single_turn_dialog_recorder_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		dataloader = FakeDataLoader()
		post_key, reference_key, gen_key = self.default_keywords \
			if argument == 'default' else ('pk', 'rk', 'gk')
		data = dataloader.get_data(post_key=post_key, reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'),
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			sr = SingleTurnDialogRecorder(dataloader)
		else:
			sr = SingleTurnDialogRecorder(dataloader, post_key, reference_key, gen_key)

		if batch_len == 'unequal':
			data[post_key] = data[post_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				sr.forward(data)
		else:
			sr.forward(data)
			assert sr.close() == self.get_sen_from_index(dataloader, data, post_key, reference_key, \
																			gen_key)
		assert same_dict(data, _data)


multi_turn_dialog_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(['empty', 'non-empty']), "multi"),
	(zip(['empty', 'non-empty']), "multi"),
	(zip(test_turn_len), "add")
)

class TestMultiTurnDialogRecorder:
	default_context_key = 'multi_turn_context_allvocabs'
	default_ref_key = 'multi_turn_ref_allvocabs'
	default_gen_key = "multi_turn_gen"
	default_turn_len_key= "turn_length"
	default_keywords = [default_context_key, default_ref_key, default_gen_key, default_turn_len_key]
	def check(self, ans, dataloader, data, context_key=default_context_key, \
			  resp_key=default_ref_key, gen_key=default_gen_key, turn_length=default_turn_len_key):
		_ans = {'context': [], 'reference': [], 'gen': []}
		for i, context_turn in enumerate(data[context_key]):
			context_now = []
			for j, context in enumerate(context_turn):
				t = dataloader.trim_index(context[1:])
				if len(t) == 0:
					break
				context_now.append(t)
			_ans['context'].append(context_now)

		for i, resp_turn in enumerate(data[resp_key]):
			resp_now = []
			for j, resp in enumerate(resp_turn):
				t = dataloader.trim_index(resp[1:])
				if data[turn_length] is None:
					if len(t) == 0:
						break
				elif j >= data[turn_length][i]:
					break
				resp_now.append(t)
			_ans['reference'].append(resp_now)

		for i, gen_turn in enumerate(data[gen_key]):
			gen_now = []
			for j, gen in enumerate(gen_turn):
				t = dataloader.trim_index(gen)
				if data[turn_length] is None:
					if len(t) == 0:
						break
				elif j >= data[turn_length][i]:
					break
				gen_now.append(t)
			_ans['gen'].append(gen_now)

		print('_ans[\'context\']: ', _ans['context'])
		print('ans[\'context\']: ', ans['context'])
		assert len(ans['context']) == len(_ans['context'])
		assert len(ans['reference']) == len(_ans['reference'])
		assert len(ans['gen']) == len(_ans['gen'])
		for i, turn in enumerate(ans['context']):
			assert len(_ans['context'][i]) == len(turn)
		for i, turn in enumerate(ans['reference']):
			assert len(_ans['reference'][i]) == len(turn)
		for i, turn in enumerate(ans['gen']):
			assert len(_ans['gen'][i]) == len(turn)

	@pytest.mark.parametrize( \
		'argument, shape, type, batch_len, gen_len, ref_len, turn_len', multi_turn_dialog_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len, turn_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeMultiDataloader()
		context_key, reference_key, gen_key, turn_len_key = self.default_keywords \
			if argument == 'default' else ('ck', 'rk', 'gk', 'tk')
		data = dataloader.get_data(context_key=context_key, turn_len_key=turn_len_key, reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtbr = MultiTurnDialogRecorder(dataloader)
		else:
			mtbr = MultiTurnDialogRecorder(dataloader, context_key, reference_key, gen_key,
										   turn_len_key)

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtbr.forward(data)
		else:
			if turn_len == 'unequal':
				data[reference_key][0] = data[reference_key][0][1:]
				with pytest.raises(ValueError, match=r"Reference turn num \d* != gen turn num \d*."):
					mtbr.forward(data)
				return
			else:
				mtbr.forward(data)
				self.check(mtbr.close(), dataloader, \
					data, context_key, reference_key, gen_key, turn_len_key)

		assert same_dict(data, _data)


language_generation_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_gen_len), "multi"),
)


class TestLanguageGenerationRecorder():
	def get_sen_from_index(self, dataloader, data, gen_key='gen'):
		ans = []
		for sen in data[gen_key]:
			ans.append(dataloader.convert_ids_to_tokens(sen))
		return ans

	@pytest.mark.parametrize('argument, shape, type, gen_len', language_generation_test_parameter)
	def test_close(self, argument, shape, type, gen_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		dataloader = FakeDataLoader()
		gen_key = 'gen' \
			if argument == 'default' else 'gk'
		data = dataloader.get_data(gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'),
								   gen_len=gen_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			lg = LanguageGenerationRecorder(dataloader)
		else:
			lg = LanguageGenerationRecorder(dataloader, gen_key)

		lg.forward(data)
		assert lg.close()['gen'] == self.get_sen_from_index(dataloader, data, gen_key)
		assert same_dict(data, _data)


hash_value_recorder_test_parameter = generate_testcase(\
	(zip(test_argument), "multi"),
	(zip(test_hash_data), "multi")
)


class TestMetricChain():
	def test_init(self):
		mc = MetricChain()

	def test_add_metric(self):
		mc = MetricChain()
		with pytest.raises(TypeError):
			mc.add_metric([1, 2, 3])

	def test_close1(self):
		dataloader = FakeMultiDataloader()
		data = dataloader.get_data(reference_key='reference_key', reference_len_key='reference_len_key', \
								   turn_len_key='turn_len_key', gen_prob_key='gen_prob_key', \
								   gen_key='gen_key', context_key='context_key')
		pm = MultiTurnPerplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', \
									   invalid_vocab=True, full_check=True)
		perplexity = TestMultiTurnPerplexityMetric().get_perplexity( \
			data, dataloader, True, 'reference_key', 'reference_len_key', 'gen_prob_key')

		bcm = MultiTurnBleuCorpusMetric(dataloader, multi_turn_reference_allvocabs_key='reference_key', \
										multi_turn_gen_key='gen_key', turn_len_key='turn_len_key')
		bleu = TestMultiTurnBleuCorpusMetric().get_bleu(dataloader, data, 'reference_key', 'gen_key')

		_data = copy.deepcopy(data)
		mc = MetricChain()
		mc.add_metric(pm)
		mc.add_metric(bcm)
		mc.forward(data)
		res = mc.close()

		assert np.isclose(res['perplexity'], perplexity)
		assert np.isclose(res['bleu'], bleu)
		assert same_dict(data, _data)
