import { useState, useEffect } from "react";

// ─── Synthetic Dataset ───────────────────────────────────────────────────────
const DATASET = [
  // PHISHING
  { text: "Urgent: Your account has been suspended! Click http://paypa1-secure.xyz/verify to restore access immediately.", label: 1 },
  { text: "Congratulations! You won $5000. Claim your prize at http://free-prize.ru/claim now!", label: 1 },
  { text: "Dear user, verify your bank credentials at http://bankofamerica-login.tk or your account will be deleted.", label: 1 },
  { text: "ALERT: Unusual login detected. Confirm your identity: http://secure-login-update.com/account", label: 1 },
  { text: "Your PayPal account is limited. Visit http://paypal-resolution.net/fix to remove limitations.", label: 1 },
  { text: "IRS Tax Refund: You have a pending $2,400 refund. Click http://irs-refund.xyz/claim to process.", label: 1 },
  { text: "Account verification required. Log in at http://apple-id-verify.gq/signin or lose access.", label: 1 },
  { text: "Your package is held. Pay $1.99 fee at http://usps-delivery-fee.info/release to release it.", label: 1 },
  { text: "Hi, I am a Nigerian prince needing help moving $10 million. Reply with your bank details.", label: 1 },
  { text: "Your Microsoft account password expires today. Reset now: http://microsoft-password-reset.xyz", label: 1 },
  { text: "WINNER! Amazon lottery pick. Provide SSN to http://amazon-gifts-official.ru to claim iPad.", label: 1 },
  { text: "Immediate action required: Your Netflix subscription failed. Update billing: http://netflix-billing.tk", label: 1 },
  // SAFE
  { text: "Hi John, just following up on the project proposal we discussed last Tuesday. Let me know your thoughts.", label: 0 },
  { text: "Your monthly invoice #4521 is attached. Total due: $120. Payment due by end of month. Thanks!", label: 0 },
  { text: "Team standup notes from today's meeting are in the shared Google Doc. Please review before Friday.", label: 0 },
  { text: "Reminder: Performance reviews are scheduled for next week. Check your calendar for your slot.", label: 0 },
  { text: "Hello, I wanted to share the research paper we mentioned. I've attached the PDF for your review.", label: 0 },
  { text: "Your order #78432 has shipped! Estimated delivery: 3-5 business days. Track at ups.com.", label: 0 },
  { text: "Welcome to the team! Your onboarding documents are ready. IT will set up your laptop tomorrow.", label: 0 },
  { text: "Conference call scheduled for Thursday 2pm. Dial-in details in the calendar invite. See you then!", label: 0 },
  { text: "The quarterly budget report is ready for your review. I've sent it to the shared finance folder.", label: 0 },
  { text: "Happy Birthday! Hope you have a wonderful day. Let's grab coffee soon to catch up.", label: 0 },
  { text: "Your subscription renewal is coming up on June 30. No action needed if you want to continue.", label: 0 },
  { text: "Thanks for attending the webinar. Slides and recording are available at our official website.", label: 0 },
];

// ─── Feature Extraction ──────────────────────────────────────────────────────
function extractFeatures(text) {
  const lower = text.toLowerCase();
  return {
    hasUrl: /https?:\/\//.test(text) ? 1 : 0,
    hasSuspiciousTLD: /\.(xyz|ru|tk|gq|cf|ml|info|cc|pw|top)\b/.test(lower) ? 1 : 0,
    urgencyWords: (lower.match(/\b(urgent|immediately|alert|action required|expire|suspend|limited|winner|congratulations|claim)\b/g) || []).length,
    sensitiveWords: (lower.match(/\b(password|bank|ssn|credit card|verify|credentials|account|billing|login|signin)\b/g) || []).length,
    exclMarks: (text.match(/!/g) || []).length,
    capsRatio: (text.match(/[A-Z]/g) || []).length / Math.max(text.length, 1),
    wordCount: text.split(/\s+/).length,
    hasIPInUrl: /https?:\/\/\d{1,3}\.\d{1,3}/.test(text) ? 1 : 0,
    hasMismatchedDomain: /paypa1|micosoft|amaz0n|g00gle|bankofamerica-/.test(lower) ? 1 : 0,
    moneyMentioned: /\$[\d,]+|\d+ (million|thousand|dollars)/i.test(text) ? 1 : 0,
  };
}

// ─── Naive Bayes Classifier ───────────────────────────────────────────────────
class NaiveBayesClassifier {
  constructor() { this.classStats = {}; this.featureNames = []; this.trained = false; }

  train(samples) {
    const classes = [0, 1];
    this.featureNames = Object.keys(extractFeatures(""));
    this.classStats = {};

    classes.forEach(c => {
      const subset = samples.filter(s => s.label === c);
      const features = subset.map(s => extractFeatures(s.text));
      this.classStats[c] = {
        prior: subset.length / samples.length,
        means: {},
        stds: {},
        count: subset.length,
      };
      this.featureNames.forEach(f => {
        const vals = features.map(x => x[f]);
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length) + 0.01;
        this.classStats[c].means[f] = mean;
        this.classStats[c].stds[f] = std;
      });
    });
    this.trained = true;
  }

  gaussianLogProb(x, mean, std) {
    return -Math.log(std * Math.sqrt(2 * Math.PI)) - ((x - mean) ** 2) / (2 * std ** 2);
  }

  predict(text) {
    const features = extractFeatures(text);
    const scores = {};
    [0, 1].forEach(c => {
      let logProb = Math.log(this.classStats[c].prior);
      this.featureNames.forEach(f => {
        logProb += this.gaussianLogProb(features[f], this.classStats[c].means[f], this.classStats[c].stds[f]);
      });
      scores[c] = logProb;
    });
    const maxScore = Math.max(scores[0], scores[1]);
    const exp0 = Math.exp(scores[0] - maxScore);
    const exp1 = Math.exp(scores[1] - maxScore);
    const prob1 = exp1 / (exp0 + exp1);
    return { label: prob1 > 0.5 ? 1 : 0, phishingProb: prob1, features };
  }
}

// ─── Train/Test Split & Evaluate ────────────────────────────────────────────
function trainAndEvaluate() {
  const shuffled = [...DATASET].sort(() => Math.random() - 0.5);
  const splitIdx = Math.floor(shuffled.length * 0.75);
  const train = shuffled.slice(0, splitIdx);
  const test = shuffled.slice(splitIdx);

  const clf = new NaiveBayesClassifier();
  clf.train(train);

  let tp = 0, tn = 0, fp = 0, fn = 0;
  const results = test.map(s => {
    const pred = clf.predict(s.text);
    if (pred.label === 1 && s.label === 1) tp++;
    else if (pred.label === 0 && s.label === 0) tn++;
    else if (pred.label === 1 && s.label === 0) fp++;
    else fn++;
    return { ...s, predicted: pred.label, prob: pred.phishingProb };
  });

  const accuracy = ((tp + tn) / test.length) * 100;
  const precision = tp / (tp + fp + 0.001) * 100;
  const recall = tp / (tp + fn + 0.001) * 100;
  const f1 = 2 * precision * recall / (precision + recall + 0.001);

  return { clf, accuracy, precision, recall, f1, confMatrix: { tp, tn, fp, fn }, testResults: results };
}

// ─── Feature Bar ─────────────────────────────────────────────────────────────
function FeatureBar({ name, value, max = 5 }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = pct > 60 ? "#ef4444" : pct > 30 ? "#f97316" : "#22c55e";
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94a3b8", marginBottom: 2 }}>
        <span>{name.replace(/([A-Z])/g, ' $1').trim()}</span>
        <span style={{ color }}>{typeof value === "number" && value % 1 !== 0 ? (value * 100).toFixed(0) + "%" : value}</span>
      </div>
      <div style={{ height: 5, background: "#1e293b", borderRadius: 3 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 3, transition: "width 0.4s ease" }} />
      </div>
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function PhishingDetector() {
  const [model, setModel] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [inputEmail, setInputEmail] = useState("");
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("detector");
  const [training, setTraining] = useState(false);
  const [trainLog, setTrainLog] = useState([]);

  const SAMPLE_PHISHING = "URGENT: Your account is suspended! Click http://paypa1-secure.xyz/verify to restore access immediately or lose your data!";
  const SAMPLE_SAFE = "Hi, please find attached the meeting notes from yesterday's call. Let me know if you have any questions.";

  useEffect(() => { runTraining(); }, []);

  async function runTraining() {
    setTraining(true);
    setTrainLog([]);
    const steps = [
      "Loading dataset (24 samples: 12 phishing, 12 safe)...",
      "Extracting features: URLs, TLDs, urgency words, caps ratio...",
      "Splitting: 75% train / 25% test...",
      "Training Gaussian Naïve Bayes classifier...",
      "Evaluating on test set...",
      "Model ready ✓",
    ];
    for (let i = 0; i < steps.length; i++) {
      await new Promise(r => setTimeout(r, 380));
      setTrainLog(prev => [...prev, steps[i]]);
    }
    const eval_ = trainAndEvaluate();
    setModel(eval_.clf);
    setMetrics(eval_);
    setTraining(false);
  }

  function classify() {
    if (!model || !inputEmail.trim()) return;
    const pred = model.predict(inputEmail);
    setResult(pred);
  }

  const tabs = ["detector", "metrics", "dataset"];

  return (
    <div style={{ minHeight: "100vh", background: "#0a0f1a", color: "#e2e8f0", fontFamily: "'JetBrains Mono', 'Courier New', monospace", padding: "24px 16px" }}>
      {/* Header */}
      <div style={{ maxWidth: 800, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
          <div style={{ width: 36, height: 36, background: "linear-gradient(135deg,#ef4444,#7c3aed)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>🎣</div>
          <div>
            <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, letterSpacing: "-0.5px", color: "#f1f5f9" }}>PhishGuard ML</h1>
            <p style={{ margin: 0, fontSize: 11, color: "#64748b" }}>Naïve Bayes · Scikit-learn Style · 10 Features</p>
          </div>
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: training ? "#f59e0b" : "#22c55e", boxShadow: `0 0 8px ${training ? "#f59e0b" : "#22c55e"}` }} />
            <span style={{ fontSize: 11, color: "#64748b" }}>{training ? "Training..." : "Model Active"}</span>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 20, borderBottom: "1px solid #1e293b", paddingBottom: 0 }}>
          {tabs.map(t => (
            <button key={t} onClick={() => setActiveTab(t)} style={{ background: "none", border: "none", cursor: "pointer", padding: "8px 16px", fontSize: 12, fontFamily: "inherit", color: activeTab === t ? "#a78bfa" : "#64748b", borderBottom: activeTab === t ? "2px solid #a78bfa" : "2px solid transparent", transition: "all 0.2s" }}>
              {t === "detector" ? "🔍 Detector" : t === "metrics" ? "📊 Metrics" : "📂 Dataset"}
            </button>
          ))}
        </div>

        {/* ── DETECTOR TAB ── */}
        {activeTab === "detector" && (
          <div>
            {/* Training log */}
            {trainLog.length > 0 && (
              <div style={{ background: "#0d1526", border: "1px solid #1e293b", borderRadius: 8, padding: 14, marginBottom: 16, fontSize: 11 }}>
                {trainLog.map((l, i) => (
                  <div key={i} style={{ color: i === trainLog.length - 1 ? "#22c55e" : "#64748b", marginBottom: 2 }}>
                    <span style={{ color: "#334155" }}>$&nbsp;</span>{l}
                  </div>
                ))}
              </div>
            )}

            {/* Input */}
            <div style={{ background: "#0d1526", border: "1px solid #1e293b", borderRadius: 10, padding: 16, marginBottom: 14 }}>
              <label style={{ fontSize: 11, color: "#64748b", display: "block", marginBottom: 8 }}>EMAIL CONTENT</label>
              <textarea
                value={inputEmail}
                onChange={e => { setInputEmail(e.target.value); setResult(null); }}
                placeholder="Paste email text here..."
                style={{ width: "100%", minHeight: 100, background: "#0a0f1a", border: "1px solid #1e293b", borderRadius: 6, color: "#e2e8f0", fontSize: 12, fontFamily: "inherit", padding: 10, resize: "vertical", outline: "none", boxSizing: "border-box" }}
              />
              <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
                <button onClick={() => { setInputEmail(SAMPLE_PHISHING); setResult(null); }} style={{ fontSize: 11, padding: "5px 12px", background: "#2d1515", border: "1px solid #7f1d1d", borderRadius: 5, color: "#fca5a5", cursor: "pointer", fontFamily: "inherit" }}>
                  Try Phishing Sample
                </button>
                <button onClick={() => { setInputEmail(SAMPLE_SAFE); setResult(null); }} style={{ fontSize: 11, padding: "5px 12px", background: "#0f2920", border: "1px solid #14532d", borderRadius: 5, color: "#86efac", cursor: "pointer", fontFamily: "inherit" }}>
                  Try Safe Sample
                </button>
                <button onClick={classify} disabled={!model || !inputEmail.trim()} style={{ marginLeft: "auto", fontSize: 12, padding: "6px 20px", background: model ? "linear-gradient(135deg,#7c3aed,#4f46e5)" : "#1e293b", border: "none", borderRadius: 6, color: model ? "#fff" : "#475569", cursor: model ? "pointer" : "not-allowed", fontFamily: "inherit", fontWeight: 600, letterSpacing: "0.5px" }}>
                  ANALYZE →
                </button>
              </div>
            </div>

            {/* Result */}
            {result && (
              <div style={{ background: result.label === 1 ? "#160a0a" : "#0a160d", border: `1px solid ${result.label === 1 ? "#7f1d1d" : "#14532d"}`, borderRadius: 10, padding: 18, marginBottom: 14 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                  <div style={{ fontSize: 32 }}>{result.label === 1 ? "🚨" : "✅"}</div>
                  <div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: result.label === 1 ? "#ef4444" : "#22c55e" }}>
                      {result.label === 1 ? "PHISHING DETECTED" : "EMAIL IS SAFE"}
                    </div>
                    <div style={{ fontSize: 12, color: "#64748b", marginTop: 2 }}>
                      Confidence: <span style={{ color: result.label === 1 ? "#ef4444" : "#22c55e", fontWeight: 600 }}>
                        {result.label === 1 ? (result.phishingProb * 100).toFixed(1) : ((1 - result.phishingProb) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div style={{ marginLeft: "auto", textAlign: "right" }}>
                    <div style={{ fontSize: 10, color: "#475569", marginBottom: 4 }}>PHISHING PROBABILITY</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: result.label === 1 ? "#ef4444" : "#22c55e" }}>
                      {(result.phishingProb * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
                {/* Probability bar */}
                <div style={{ height: 8, background: "#1e293b", borderRadius: 4, marginBottom: 16, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${result.phishingProb * 100}%`, background: `linear-gradient(90deg, #22c55e, #ef4444)`, borderRadius: 4, transition: "width 0.6s ease" }} />
                </div>
                {/* Feature breakdown */}
                <div style={{ fontSize: 11, color: "#64748b", marginBottom: 8 }}>FEATURE ANALYSIS</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 24px" }}>
                  <FeatureBar name="Has URL" value={result.features.hasUrl} max={1} />
                  <FeatureBar name="Suspicious TLD" value={result.features.hasSuspiciousTLD} max={1} />
                  <FeatureBar name="Urgency Words" value={result.features.urgencyWords} max={5} />
                  <FeatureBar name="Sensitive Words" value={result.features.sensitiveWords} max={5} />
                  <FeatureBar name="Exclamation Marks" value={result.exclMarks} max={4} />
                  <FeatureBar name="Caps Ratio" value={result.features.capsRatio} max={1} />
                  <FeatureBar name="Money Mentioned" value={result.features.moneyMentioned} max={1} />
                  <FeatureBar name="Mismatched Domain" value={result.features.hasMismatchedDomain} max={1} />
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── METRICS TAB ── */}
        {activeTab === "metrics" && metrics && (
          <div>
            {/* Score cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 16 }}>
              {[
                { label: "Accuracy", value: metrics.accuracy, color: "#a78bfa" },
                { label: "Precision", value: metrics.precision, color: "#38bdf8" },
                { label: "Recall", value: metrics.recall, color: "#34d399" },
                { label: "F1 Score", value: metrics.f1, color: "#fb923c" },
              ].map(m => (
                <div key={m.label} style={{ background: "#0d1526", border: "1px solid #1e293b", borderRadius: 8, padding: "12px 10px", textAlign: "center" }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: m.color }}>{m.value.toFixed(1)}%</div>
                  <div style={{ fontSize: 10, color: "#64748b", marginTop: 3 }}>{m.label}</div>
                </div>
              ))}
            </div>

            {/* Confusion Matrix */}
            <div style={{ background: "#0d1526", border: "1px solid #1e293b", borderRadius: 10, padding: 18, marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: "#64748b", marginBottom: 14 }}>CONFUSION MATRIX</div>
              <div style={{ display: "grid", gridTemplateColumns: "80px 1fr 1fr", gap: 6, fontSize: 11 }}>
                <div />
                <div style={{ textAlign: "center", color: "#64748b", paddingBottom: 4 }}>Pred: Safe</div>
                <div style={{ textAlign: "center", color: "#64748b", paddingBottom: 4 }}>Pred: Phish</div>
                <div style={{ color: "#64748b", display: "flex", alignItems: "center" }}>Actual: Safe</div>
                <div style={{ background: "#0a2213", border: "1px solid #14532d", borderRadius: 6, padding: "12px 0", textAlign: "center" }}>
                  <div style={{ fontSize: 22, fontWeight: 700, color: "#22c55e" }}>{metrics.confMatrix.tn}</div>
                  <div style={{ fontSize: 10, color: "#64748b" }}>True Neg</div>
                </div>
                <div style={{ background: "#1a0a0a", border: "1px solid #7f1d1d", borderRadius: 6, padding: "12px 0", textAlign: "center" }}>
                  <div style={{ fontSize: 22, fontWeight: 700, color: "#ef4444" }}>{metrics.confMatrix.fp}</div>
                  <div style={{ fontSize: 10, color: "#64748b" }}>False Pos</div>
                </div>
                <div style={{ color: "#64748b", display: "flex", alignItems: "center" }}>Actual: Phish</div>
                <div style={{ background: "#1a0a0a", border: "1px solid #7f1d1d", borderRadius: 6, padding: "12px 0", textAlign: "center" }}>
                  <div style={{ fontSize: 22, fontWeight: 700, color: "#ef4444" }}>{metrics.confMatrix.fn}</div>
                  <div style={{ fontSize: 10, color: "#64748b" }}>False Neg</div>
                </div>
                <div style={{ background: "#0a2213", border: "1px solid #14532d", borderRadius: 6, padding: "12px 0", textAlign: "center" }}>
                  <div style={{ fontSize: 22, fontWeight: 700, color: "#22c55e" }}>{metrics.confMatrix.tp}</div>
                  <div style={{ fontSize: 10, color: "#64748b" }}>True Pos</div>
                </div>
              </div>
            </div>

            {/* Test results */}
            <div style={{ background: "#0d1526", border: "1px solid #1e293b", borderRadius: 10, padding: 16 }}>
              <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12 }}>TEST SET PREDICTIONS ({metrics.testResults.length} samples)</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {metrics.testResults.map((r, i) => (
                  <div key={i} style={{ display: "flex", gap: 10, alignItems: "center", padding: "8px 10px", background: r.predicted === r.label ? "#0a1520" : "#1a0810", borderRadius: 6, border: `1px solid ${r.predicted === r.label ? "#1e293b" : "#4c0519"}` }}>
                    <span style={{ fontSize: 14 }}>{r.predicted === 1 ? "🎣" : "✅"}</span>
                    <span style={{ fontSize: 10, color: "#94a3b8", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.text.slice(0, 70)}…</span>
                    <span style={{ fontSize: 10, fontWeight: 600, color: r.predicted === r.label ? "#64748b" : "#f43f5e", whiteSpace: "nowrap" }}>
                      {r.predicted === r.label ? "✓ Correct" : "✗ Wrong"}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <button onClick={runTraining} style={{ marginTop: 12, width: "100%", padding: "10px 0", background: "#1e293b", border: "1px solid #334155", borderRadius: 8, color: "#94a3b8", cursor: "pointer", fontFamily: "inherit", fontSize: 12 }}>
              🔄 Retrain Model (Random Split)
            </button>
          </div>
        )}

        {/* ── DATASET TAB ── */}
        {activeTab === "dataset" && (
          <div>
            <div style={{ background: "#0d1526", border: "1px solid #1e293b", borderRadius: 10, padding: 14, marginBottom: 14, fontSize: 11, color: "#64748b", lineHeight: 1.7 }}>
              <strong style={{ color: "#94a3b8" }}>Model Architecture:</strong> Gaussian Naïve Bayes<br />
              <strong style={{ color: "#94a3b8" }}>Features (10):</strong> URL presence, suspicious TLD, urgency words, sensitive keywords, exclamation marks, caps ratio, word count, IP in URL, domain mismatch, money mentioned<br />
              <strong style={{ color: "#94a3b8" }}>Dataset:</strong> 24 labeled samples (12 phishing, 12 safe)<br />
              <strong style={{ color: "#94a3b8" }}>Split:</strong> 75% train / 25% test (random shuffle each run)
            </div>
            {[1, 0].map(label => (
              <div key={label} style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 11, color: label === 1 ? "#ef4444" : "#22c55e", marginBottom: 8 }}>
                  {label === 1 ? "🎣 PHISHING SAMPLES" : "✅ SAFE SAMPLES"} ({DATASET.filter(d => d.label === label).length})
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {DATASET.filter(d => d.label === label).map((d, i) => (
                    <div key={i} style={{ background: "#0a0f1a", border: `1px solid ${label === 1 ? "#3d1a1a" : "#1a3d2a"}`, borderRadius: 6, padding: "10px 12px", fontSize: 11, color: "#94a3b8", lineHeight: 1.5, cursor: "pointer" }}
                      onClick={() => { setInputEmail(d.text); setActiveTab("detector"); setResult(null); }}>
                      {d.text}
                      <span style={{ marginLeft: 8, fontSize: 10, color: "#475569" }}>(click to test)</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        <div style={{ marginTop: 20, fontSize: 10, color: "#334155", textAlign: "center" }}>
          Gaussian Naïve Bayes · 10 handcrafted features · runs entirely in-browser
        </div>
      </div>
    </div>
  );
}
