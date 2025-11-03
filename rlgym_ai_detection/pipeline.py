import re, numpy as np, time, logging
from .normalize_segment import normalize
from .features import TextFeaturizer
from .features_extra import tokenize_words, tokenize_sentences, function_word_profile, coherence_graph_metrics, FUNCTION_WORDS_EN
from .deep_ensemble import HeteroEnsemble
from .ood import MahalanobisOOD
from .ood_extra import fit_iforest, iforest_score
from .policies import load_policy
from .conformal_mondrian import MondrianConformal
from .utils import detect_lang, get_logger

def default_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
        return lambda texts: model.encode(texts, normalize_embeddings=True)
    except Exception:
        import hashlib, numpy as np, re
        def _emb(texts):
            out = []
            for t in texts:
                toks = re.findall(r"\w+", t.lower())
                dim = 256
                v = np.zeros(dim, dtype=float)
                for tok in toks:
                    h = int(hashlib.sha1(tok.encode()).hexdigest(),16) % dim
                    v[h]+=1.0
                if v.sum()>0: v = v/np.linalg.norm(v)
                out.append(v)
            return np.vstack(out) if out else np.zeros((0,256))
        return _emb

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split
except Exception:
    TfidfVectorizer = LogisticRegression = GradientBoostingClassifier = MLPClassifier = IsotonicRegression = train_test_split = None

class DetectionPipeline:
    """AI-generated content detection pipeline with ensemble learning, OOD detection, and conformal prediction.
    
    This pipeline combines multiple detection strategies:
    - Heterogeneous ensemble (TF-IDF, dense features, semantic embeddings)
    - Out-of-distribution detection (Mahalanobis + IsolationForest)
    - Isotonic calibration for probability estimates
    - Mondrian conformal prediction for uncertainty quantification
    
    Args:
        policy: Policy configuration dict with tiers and thresholds. If None, loads default policy.
        embed_fn: Function to generate text embeddings. If None, uses default_embedder().
        logger: Logger instance. If None, creates default logger.
    
    Example:
        >>> pipe = DetectionPipeline()
        >>> pipe.fit(texts, labels, langs=["EN", "pt-BR", ...])
        >>> report = pipe.predict_one("Your text here", risk_tier="medium")
        >>> print(report["decision"])  # "likely_human", "likely_llm", or "abstain"
    """
    def __init__(self, policy: dict | None = None, embed_fn=None, logger: logging.Logger | None = None):
        self.policy = policy or load_policy()
        self.embed = embed_fn or default_embedder()
        self.fe_basic = TextFeaturizer()
        self.ood_maha = MahalanobisOOD()
        self.ood_iforest = None
        self.conformal = MondrianConformal(q=0.1)
        self.learners = None
        self.calib = None
        self.model_ready = False
        self.logger = logger or get_logger()

    def _tier_cfg(self, risk_tier: str):
        return (self.policy.get("tiers") or {}).get(risk_tier or "medium") or {}

    def _extra_dense(self, texts, sent_emb_fn):
        mats = []
        for t in texts:
            sents = tokenize_sentences(t)
            if not sents:
                mats.append([1.0,0.0] + [0.0]*len(FUNCTION_WORDS_EN))
                continue
            E = sent_emb_fn(sents)
            density, gap = coherence_graph_metrics(E)
            fw = function_word_profile(tokenize_words(t), FUNCTION_WORDS_EN).tolist()
            mats.append([density, gap] + fw)
        return np.array(mats, dtype=float)

    def _meta(self, text: str, lang: str):
        n = len(text)
        L = "len:short" if n<200 else ("len:medium" if n<1200 else "len:long")
        has_code = "code:yes" if re.search(r"```", text) else "code:no"
        return [f"lang:{lang}", L, has_code]

    def fit(self, texts, y, langs=None):
        """Train the detection pipeline on labeled data.
        
        Args:
            texts: List of text samples (strings)
            y: List of labels (0=human, 1=LLM)
            langs: Optional list of language codes (e.g., ["EN", "pt-BR"]). Defaults to "EN".
        
        Returns:
            self: Returns self for method chaining
        
        Raises:
            ValueError: If inputs are invalid (empty, mismatched lengths, invalid labels, etc.)
            RuntimeError: If scikit-learn is not installed
        
        Example:
            >>> texts = ["Human text", "LLM-generated text"]
            >>> labels = [0, 1]
            >>> pipe.fit(texts, labels)
        """
        # Validate dependencies
        if not (TfidfVectorizer and LogisticRegression and GradientBoostingClassifier and MLPClassifier and IsotonicRegression and train_test_split):
            raise RuntimeError("Training requires scikit-learn. Install extras: pip install rlgym-ai-detection[full]")
        
        # Validate inputs
        if not texts:
            raise ValueError("texts cannot be empty")
        if not y:
            raise ValueError("y cannot be empty")
        if len(texts) != len(y):
            raise ValueError(f"texts ({len(texts)}) and y ({len(y)}) must have same length")
        if len(texts) < 4:
            raise ValueError(f"Need at least 4 samples for training, got {len(texts)}")
        
        # Validate labels
        y_array = np.asarray(y)
        unique_labels = np.unique(y_array)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(f"y must contain only 0 and 1, got {unique_labels}")
        if len(unique_labels) < 2:
            raise ValueError("y must contain both classes (0 and 1)")
        
        # Validate langs
        langs = langs or ["EN"]*len(texts)
        if len(langs) != len(texts):
            raise ValueError(f"langs ({len(langs)}) must match texts ({len(texts)}) length")
        
        texts = [normalize(t) for t in texts]
        # basic + extra features
        Xs, Xd_basic = self.fe_basic.fit(texts).transform(texts)
        Xsem_doc = self.embed(texts)
        XD_extra = self._extra_dense(texts, self.embed)
        XD_all = np.hstack([Xd_basic, XD_extra])
        # learners
        vec = TfidfVectorizer(ngram_range=(1,3), min_df=3, max_df=0.95)
        clf1 = LogisticRegression(max_iter=500, class_weight='balanced')
        clf2 = GradientBoostingClassifier()
        clf3 = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200)
        # Prepare learners dicts
        learners = [
            {"vec": vec, "clf": clf1, "space":"sparse"},
            {"vec": None, "clf": clf2, "space":"dense", "Xd": XD_all, "Xd_test": None},
            {"vec": None, "clf": clf3, "space":"sem", "Xsem": Xsem_doc, "Xsem_test": None},
        ]
        # Fit ensemble
        self.learners = learners
        self.learners[1]["Xd"] = XD_all
        self.learners[2]["Xsem"] = Xsem_doc
        # Sparse fit is inside ensemble.fit
        H = HeteroEnsemble(self.learners).fit(texts, y)
        XA = self.learners[0]["vec"].transform(texts)
        p1 = self.learners[0]["clf"].predict_proba(XA)[:,1]
        p2 = self.learners[1]["clf"].predict_proba(XD_all)[:,1]
        p3 = self.learners[2]["clf"].predict_proba(Xsem_doc)[:,1]
        p_stack = (p1+p2+p3)/3.0
        # Held-out calibration split
        Xc, Xh, yc, yh = train_test_split(p_stack.reshape(-1,1), np.asarray(y), test_size=0.2, random_state=42, stratify=np.asarray(y))
        self.calib = IsotonicRegression(out_of_bounds="clip").fit(Xc.ravel(), yc)
        # OOD fit
        X_all_for_ood = np.hstack([XA.toarray(), XD_all, Xsem_doc])
        self.ood_maha.fit(X_all_for_ood)
        from .ood_extra import fit_iforest
        self.ood_iforest = fit_iforest(X_all_for_ood, contamination=0.08)
        # Conformal
        metas = [self._meta(t, lang=l) for t,l in zip(texts, langs)]
        self.conformal.fit(self.calib.predict(p_stack), np.asarray(y), metas)
        self.model_ready = True
        self.logger.info("fit: n=%d, calib_holdout=%d", len(texts), len(yc))
        return self

    def predict_one(self, text: str, risk_tier: str = "medium", lang: str | None = None):
        """Predict whether a text is AI-generated.
        
        Args:
            text: Text to analyze (string)
            risk_tier: Risk tolerance level - "low", "medium", or "high".
                      Higher tiers use stricter thresholds and are more likely to abstain.
            lang: Language code (e.g., "EN", "pt-BR"). If None, auto-detects.
        
        Returns:
            dict: Report with keys:
                - decision: "likely_human", "likely_llm", or "abstain"
                - prob_llm: Calibrated probability of being LLM-generated (0-1)
                - ood_scores: Out-of-distribution scores (mahalanobis, iforest)
                - confidence_band: Abstention band for this risk tier
                - signals: Structural, stylistic, and lexical features
                - version: Model version
                - policy_version: Policy version
        
        Raises:
            RuntimeError: If pipeline not fitted
            ValueError: If text is empty or risk_tier is invalid
            TypeError: If text is not a string
        
        Example:
            >>> report = pipe.predict_one(
            ...     "System prompt: Act as SRE. Task: write runbook.",
            ...     risk_tier="high"
            ... )
            >>> print(f"{report['decision']}: {report['prob_llm']:.2f}")
        """
        # Validate model state
        if not self.model_ready:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        # Validate inputs
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text or not text.strip():
            raise ValueError("text cannot be empty or whitespace")
        if risk_tier not in ["low", "medium", "high"]:
            raise ValueError(f"risk_tier must be 'low', 'medium', or 'high', got '{risk_tier}'")
        
        t = normalize(text)
        if not lang: lang = detect_lang(t)
        self.logger.info("predict_one: tier=%s lang=%s len=%d", risk_tier, lang, len(t))
        # features
        XA = self.learners[0]["vec"].transform([t])
        Xsem = self.embed([t])
        from .features import TextFeaturizer
        tf = TextFeaturizer()
        _, Xd_basic = tf.transform([t])
        XD_extra = self._extra_dense([t], self.embed)
        XD_all = np.hstack([Xd_basic, XD_extra])
        # proba (stack + calib)
        p1 = self.learners[0]["clf"].predict_proba(XA)[:,1]
        p2 = self.learners[1]["clf"].predict_proba(XD_all)[:,1]
        p3 = self.learners[2]["clf"].predict_proba(Xsem)[:,1]
        p_raw = (p1+p2+p3)/3.0
        p = float(self.calib.predict(p_raw)[0])
        # OOD
        X_all_for_ood = np.hstack([XA.toarray(), XD_all, Xsem])
        ood_maha = float(self.ood_maha.score(X_all_for_ood)[0])
        from .ood_extra import iforest_score
        ood_if = float(iforest_score(self.ood_iforest, X_all_for_ood)[0])
        # Policy & conformal
        tier = self._tier_cfg(risk_tier)
        lo, hi = tier.get("abstain_band",[0.4,0.6])
        ood_thr = float(tier.get("ood_threshold",1.2))
        meta = self._meta(t, lang)
        abst_conformal = self.conformal.abstain(p, meta)
        decision = "likely_human" if p < lo else ("likely_llm" if p > hi else "abstain")
        if abst_conformal or (ood_maha>ood_thr) or (ood_if>ood_thr):
            decision = "abstain"
        # signals
        tokens = re.findall(r"\w+", t.lower())
        ttr = len(set(tokens))/max(1,len(tokens))
        code_fences = len(re.findall(r"```", t))
        list_markers = len(re.findall(r"^[-*+]\s", t, flags=re.M))
        density, gap = float(XD_extra[0,0]), float(XD_extra[0,1])
        fw = function_word_profile(tokens).tolist()
        report = {
            "version":"1.4.0",
            "decision": decision,
            "prob_llm": p,
            "ood_scores": {"mahalanobis": ood_maha, "iforest": ood_if},
            "confidence_band": f"{lo:.2f}-{hi:.2f}",
            "model_version": "detector-ensemble-1.4.0",
            "policy_version": str(self.policy.get("policy_version")),
            "signals": {
                "struct": {"length_chars": len(t), "code_fences": code_fences, "list_markers": list_markers},
                "style": {"coherence_density": density, "coherence_gap": gap},
                "lex": {"ttr": ttr, "function_words_var": float(np.var(fw)), "function_words_mean": float(np.mean(fw))},
                "sem": {},
                "telemetry": {}
            },
            "policy_notes": "Conformal/OOD abstention" if decision=="abstain" else ""
        }
        return report
