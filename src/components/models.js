export function createModels() {
  return `
    <section class="models" id="models">
      <div class="container">
        <div class="section-header animate-on-scroll">
          <h2>Our AI Model Arsenal</h2>
          <p>Each model is specifically designed and trained for different aspects of market analysis and prediction.</p>
        </div>
        <div class="models-grid">
          <div class="model-card animate-on-scroll">
            <div class="model-header">
              <div class="model-icon">
                <i class="fas fa-network-wired"></i>
              </div>
              <h3>Generative Adversarial Networks (GANs)</h3>
            </div>
            <p>Generate synthetic market data and identify hidden patterns in price movements through adversarial training processes.</p>
            <ul class="model-features">
              <li>Market scenario simulation</li>
              <li>Pattern generation and recognition</li>
              <li>Anomaly detection in trading data</li>
              <li>Risk scenario modeling</li>
            </ul>
          </div>
          <div class="model-card animate-on-scroll">
            <div class="model-header">
              <div class="model-icon">
                <i class="fas fa-compress-arrows-alt"></i>
              </div>
              <h3>Variational Autoencoders (VAEs)</h3>
            </div>
            <p>Compress complex market data into meaningful representations and generate probabilistic market forecasts.</p>
            <ul class="model-features">
              <li>Dimensionality reduction</li>
              <li>Market state compression</li>
              <li>Probabilistic forecasting</li>
              <li>Feature extraction from noise</li>
            </ul>
          </div>
          <div class="model-card animate-on-scroll">
            <div class="model-header">
              <div class="model-icon">
                <i class="fas fa-eye"></i>
              </div>
              <h3>Convolutional Neural Networks (CNNs)</h3>
            </div>
            <p>Analyze chart patterns and technical indicators with computer vision techniques adapted for financial data.</p>
            <ul class="model-features">
              <li>Chart pattern recognition</li>
              <li>Technical indicator analysis</li>
              <li>Candlestick pattern detection</li>
              <li>Multi-timeframe analysis</li>
            </ul>
          </div>
          <div class="model-card animate-on-scroll">
            <div class="model-header">
              <div class="model-icon">
                <i class="fas fa-project-diagram"></i>
              </div>
              <h3>Multi-Head GANs (MHGANs)</h3>
            </div>
            <p>Advanced GAN architecture with multiple discriminators for enhanced market prediction accuracy and stability.</p>
            <ul class="model-features">
              <li>Multi-perspective analysis</li>
              <li>Enhanced stability</li>
              <li>Reduced mode collapse</li>
              <li>Improved convergence</li>
            </ul>
          </div>
          <div class="model-card animate-on-scroll">
            <div class="model-header">
              <div class="model-icon">
                <i class="fas fa-info-circle"></i>
              </div>
              <h3>Information GANs (InfoGANs)</h3>
            </div>
            <p>Discover interpretable market factors and latent variables that drive price movements in financial markets.</p>
            <ul class="model-features">
              <li>Interpretable representations</li>
              <li>Factor discovery</li>
              <li>Disentangled learning</li>
              <li>Market driver identification</li>
            </ul>
          </div>
          <div class="model-card animate-on-scroll">
            <div class="model-header">
              <div class="model-icon">
                <i class="fas fa-calculator"></i>
              </div>
              <h3>Bayesian Networks</h3>
            </div>
            <p>Probabilistic reasoning and uncertainty quantification for robust trading decisions under market volatility.</p>
            <ul class="model-features">
              <li>Uncertainty quantification</li>
              <li>Probabilistic inference</li>
              <li>Causal relationship modeling</li>
              <li>Risk-aware predictions</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  `
}