from string import punctuation

HPARAMS_REGISTRY = {}
DEFAULTS = {}

class Hyperparams(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value
        
def setup_hparams(hparam_set_names, kwargs):
    H = Hyperparams()
    if not isinstance(hparam_set_names, tuple):
        hparam_set_names = hparam_set_names.split(",")
    hparam_sets = [HPARAMS_REGISTRY[x] for x in hparam_set_names if x] + [kwargs]
    
    for k, v in DEFAULTS.items():
        H.update(v)
        
    for hps in hparam_sets:
#         create a mini version of the model to add to defaults
#         for k in hps:
#             if k not in H:
#                 raise ValueError(f"{k} not in default args")
        H.update(**hps)
    H.update(**kwargs)
    return H        

embedding = Hyperparams(
    pretrained_embed=False,
    padding_id=0,
    embed_in=-1,
    embed_dim=200,
    
)
HPARAMS_REGISTRY['embedding'] = embedding

cnn = Hyperparams(
    conv_in=1,
    conv_out=100,
    kernel_sizes=[3, 4, 5],
    stride=1
)
HPARAMS_REGISTRY['cnn'] = cnn

bilstm = Hyperparams(
    bidrectional=True,
    hidden_dim=200,
    num_layers=2,
    lstm_dropout=0.1
)
HPARAMS_REGISTRY['bilstm'] = bilstm

linear = Hyperparams(
    linear_in=-1,
    linear_out=1,
    linear_dropout=0.5
)
HPARAMS_REGISTRY['linear'] = linear

hps_data = Hyperparams(
    data_path_wh_popular='data/raw',
    use_all_data=False,
    keep_keys=['text', 'title', 'domain_rank', 'performance_score', 'site', 'social', 'url'],
    key_order=['title', 'sanitized_title', 'text', 'url', 'site', 'domain_rank', 'engagement_scores'],
    banned = ['news', 'cnn', 'abc', '.com', 'wsj', 'times', 'india', 'tribune', 'nbc', 'fox', 'daily', 'mirror', 'espn', 'post', 'fox', 'usa'],
    allowed = ['á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ'],
    punct=f'{punctuation}–\'\"‘’…«—-”“£',
    score='original',
    indent=4,
    max_count=1000,
    comment_weight=5,
    take_log=True,
    min_len=3,
    seq_length=25,
    split_frac=0.8,
    batch_size=10
)
DEFAULTS['data'] = hps_data

hps_save = Hyperparams(
    save_path='ckpts'
)
DEFAULTS['save'] = hps_save

hps_opt = Hyperparams(
    cuda=False,
    adam=True,
    sgd=False,    
    epochs=1000,
    lr=0.0003,
    clip=1,
)
DEFAULTS['opt'] = hps_opt