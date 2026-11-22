"""Wrapper-e pentru TSFM-uri: toate întorc forecast(batch_ctx, h) -> (lot, orizont, 3)."""

import numpy as np


def make_chronos(model_id="amazon/chronos-bolt-small", device="cuda", dtype="float32"):
    """Întoarce o funcție forecast(batch_ctx, h) -> cuantile (lot, orizont, 3).

    Ultima axă = cuantilele [0.1, 0.5, 0.9]; split_q() le desparte.

    Testat cu chronos-forecasting 2.3.2 + transformers 5.23:
      - predict_quantiles primește seria în `inputs=` (nu `context=`);
      - dtype se dă cu `dtype=` (transformers 5 a redenumit `torch_dtype`).
    float32 implicit: metrica e o DIFERENȚĂ între două prognoze aproape
    identice, iar rotunjirile din bfloat16 ar adăuga zgomot fals în `control`.
    """
    import torch
    from chronos import BaseChronosPipeline, Chronos2Pipeline

    # from_pretrained citește config-ul modelului și alege singur clasa
    # potrivită: ChronosBoltPipeline sau Chronos2Pipeline
    pipe = BaseChronosPipeline.from_pretrained(
        model_id, device_map=device, dtype=getattr(torch, dtype)
    )
    q = [0.1, 0.5, 0.9]

    def forecast_bolt(batch_ctx, h):
        # tensor 2D (lot, timp); pipeline-ul îl mută singur pe GPU
        ctx = torch.tensor(np.asarray(batch_ctx), dtype=torch.float32)
        quantiles, mean = pipe.predict_quantiles(
            inputs=ctx, prediction_length=h, quantile_levels=q
        )
        # quantiles: (lot, orizont, 3) în fp32 pe CPU: q0.1, mediana, q0.9
        return quantiles.numpy()

    def forecast_chronos2(batch_ctx, h):
        # Chronos-2 vrea 3D: (lot, variabile, timp); noi avem o singură variabilă
        ctx = torch.tensor(np.asarray(batch_ctx), dtype=torch.float32)[:, None, :]
        quantiles, mean = pipe.predict_quantiles(
            inputs=ctx, prediction_length=h, quantile_levels=q,
            cross_learning=False,  # seriile din lot NU își împrumută informație
        )
        # listă cu câte un tensor (1, orizont, 3) per serie -> (lot, orizont, 3)
        return np.stack([qt[0].float().cpu().numpy() for qt in quantiles])

    return forecast_chronos2 if isinstance(pipe, Chronos2Pipeline) else forecast_bolt


def make_chronos_t5(model_id="amazon/chronos-t5-small", device="cuda", num_samples=20, seed=0):
    """Chronos original (T5): valori cuantizate în tokeni, prognoză prin EȘANTIONARE.

    Spre deosebire de Bolt, fiecare apel trage num_samples traiectorii
    aleatoare; mediana și cuantilele vin din aceste eșantioane. Ca rezultatul
    să fie reproductibil, resetăm seed-ul la FIECARE apel: geamănul curat și
    cel cu anomalie primesc astfel aceeași secvență aleatoare, iar o parte
    din zgomotul de eșantionare se anulează în diferență.
    Ieșire: (lot, orizont, 3) = q0.1, mediana, q0.9, ca la make_chronos."""
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=device, dtype=torch.float32)

    def forecast(batch_ctx, h, seed=seed, num_samples=num_samples):
        # serii × eșantioane generate deodată: T5 copiază contextul codificat
        # pentru fiecare eșantion; 4000 nu mai încap în 16 GB, 2000 da
        chunk = max(1, 1500 // num_samples)
        torch.manual_seed(seed)
        ctx = torch.tensor(np.asarray(batch_ctx), dtype=torch.float32)
        outs = []
        for i in range(0, len(ctx), chunk):
            q, _ = pipe.predict_quantiles(inputs=ctx[i:i + chunk], prediction_length=h,
                                          quantile_levels=[0.1, 0.5, 0.9],
                                          num_samples=num_samples)
            outs.append(q.numpy())
        return np.concatenate(outs)

    return forecast


def make_moirai(model_id="Salesforce/moirai-2.0-R-small", device="cuda", batch_size=256):
    """Moirai 2.0 (Salesforce), ieșire (lot, orizont, 3) ca la celelalte modele.

    ATENȚIE: rulează doar în .venv-moirai (uni2ts cere numpy 1.26 / torch<2.5,
    incompatibile cu mediul principal; vezi NOTES.md). Ponderile sunt sub
    CC BY-NC 4.0. Valorile NaN sunt completate de uni2ts cu media cauzală,
    iar modelul primește și masca valorilor observate."""
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    module = Moirai2Module.from_pretrained(model_id)
    model = Moirai2Forecast(module=module, prediction_length=1, context_length=1,
                            target_dim=1, feat_dynamic_real_dim=0,
                            past_feat_dynamic_real_dim=0).to(device).eval()
    levels = [round(float(x), 3) for x in module.quantile_levels]
    idx = [levels.index(0.1), levels.index(0.5), levels.index(0.9)]

    def forecast(batch_ctx, h):
        batch_ctx = np.asarray(batch_ctx, dtype=np.float32)
        outs = []
        with model.hparams_context(prediction_length=h, context_length=batch_ctx.shape[1]):
            for i in range(0, len(batch_ctx), batch_size):
                p = model.predict(past_target=[c for c in batch_ctx[i:i + batch_size]])
                outs.append(np.asarray(p).reshape(len(p), len(levels), h))   # (lot, cuantile, h)
        q = np.concatenate(outs)[:, idx, :]                                  # (lot, 3, h)
        return np.transpose(q, (0, 2, 1))                                    # (lot, h, 3)

    return forecast


def make_timesfm(model_id="google/timesfm-2.5-200m-pytorch", max_context=512, max_horizon=128):
    """TimesFM 2.5 (Google), cu aceeași ieșire ca make_chronos: (lot, orizont, 3).

    Testat cu timesfm 3.0.2. Setările din compile() sunt cele recomandate în
    README. torch_compile=False: torch.compile cere Triton, care lipsește pe
    Windows (iar fără el modelul doar rulează puțin mai încet).
    TimesFM întoarce 10 coloane: 0 = media, 1..9 = cuantilele 0.1..0.9."""
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id, torch_compile=False)
    model.compile(timesfm.ForecastConfig(
        max_context=max_context,       # contextul nostru: max 8 × 48 = 384 de pași
        max_horizon=max_horizon,       # rotunjit la multiplu de 128 de bibliotecă
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,        # nu contează: seriile noastre au și valori negative
        fix_quantile_crossing=True,
        per_core_batch_size=64,
    ))

    def forecast(batch_ctx, h):
        _, q = model.forecast(horizon=h, inputs=[np.asarray(c, dtype=np.float32) for c in batch_ctx])
        return np.asarray(q)[:, :h][..., [1, 5, 9]]     # q0.1, mediana, q0.9

    return forecast


def make_tirex(model_id="NX-AI/TiRex", device="cuda"):
    """TiRex (NX-AI, arhitectură xLSTM, nu transformer), ieșire (lot, orizont, 3).

    Testat cu tirex-ts 1.4.2. backend="torch": kernelurile CUDA dedicate cer
    pachetul xlstm, greu de compilat pe Windows; varianta torch dă aceleași
    rezultate, doar mai lent."""
    import torch
    from tirex import load_model
    model = load_model(model_id, device=device, backend="torch")
    levels = list(model.config.quantiles)
    idx = [levels.index(0.1), levels.index(0.5), levels.index(0.9)]

    def forecast(batch_ctx, h):
        ctx = torch.tensor(np.asarray(batch_ctx), dtype=torch.float32)
        q, _ = model.forecast(context=ctx, prediction_length=h, output_type="numpy")
        return np.asarray(q)[..., idx]

    return forecast


# Modelele Chronos disponibile: nume scurt -> (nume în tabel, ID Hugging Face)
CHRONOS_MODELS = {
    "small": ("chronos_bolt_small", "amazon/chronos-bolt-small"),
    "base": ("chronos_bolt_base", "amazon/chronos-bolt-base"),
    "c2": ("chronos_2", "amazon/chronos-2"),
}


# Toate TSFM-urile: nume scurt -> (nume în tabel, funcție care încarcă modelul)
TSFM_MODELS = {
    **{key: (name, lambda mid=mid: make_chronos(mid))
       for key, (name, mid) in CHRONOS_MODELS.items()},
    "timesfm": ("timesfm_2p5", make_timesfm),
    "tirex": ("tirex", make_tirex),
    "t5": ("chronos_t5_small", make_chronos_t5),
    "moirai": ("moirai_2_small", make_moirai),     # doar din .venv-moirai
}
