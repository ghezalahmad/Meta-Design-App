
import streamlit as st

st.set_page_config(
    page_title="Meta-Design Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# The adapted HTML, CSS, and JS from the user's suggestion.
landing_page_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meta-Design - Sequential Learning for Materials Discovery</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            overflow-x: hidden;
            background: #0a0e27;
            color: #fff;
        }

        /* Make Streamlit's header invisible */
        header[data-testid="stHeader"] {
            display: none;
        }

        /* Hide the sidebar and its toggle button */
        div[data-testid="stSidebar"] {
            display: none;
        }

        /* Main content styling */
        .main > div {
            padding: 0;
        }

        /* Navigation */
        nav {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            padding: 1.5rem 5%;
            background: rgba(10, 14, 39, 0.8);
            backdrop-filter: blur(10px);
            z-index: 1000;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .nav-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .nav-links {
            display: flex;
            gap: 2rem;
            list-style: none;
        }

        .nav-links a {
            color: #fff;
            text-decoration: none;
            transition: color 0.3s;
            font-weight: 500;
        }

        .nav-links a:hover {
            color: #667eea;
        }

        /* Hero Section */
        .hero {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            padding: 8rem 5% 4rem;
            overflow: hidden;
        }

        .hero::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.15) 0%, transparent 50%),
                        radial-gradient(circle at 80% 80%, rgba(118, 75, 162, 0.15) 0%, transparent 50%);
        }

        .hero-content {
            max-width: 1400px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4rem;
            align-items: center;
            position: relative;
            z-index: 1;
        }

        .hero-text h1 {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 1.5rem;
            line-height: 1.2;
            background: linear-gradient(135deg, #fff 0%, #667eea 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-text p {
            font-size: 1.25rem;
            color: #a0aec0;
            margin-bottom: 2rem;
            line-height: 1.6;
        }

        .cta-buttons {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .btn {
            padding: 1rem 2rem;
            border-radius: 12px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s;
            border: none;
            cursor: pointer;
            font-size: 1rem;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }

        .hero-visual {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .cube-container {
            width: 400px;
            height: 400px;
            perspective: 1000px;
            position: relative;
        }

        .floating-elements {
            position: absolute;
            width: 100%;
            height: 100%;
        }

        .float-item {
            position: absolute;
            width: 60px;
            height: 60px;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            animation: float 3s ease-in-out infinite;
        }

        .float-item:nth-child(1) { top: 10%; left: 10%; animation-delay: 0s; }
        .float-item:nth-child(2) { top: 20%; right: 15%; animation-delay: 0.5s; }
        .float-item:nth-child(3) { bottom: 25%; left: 5%; animation-delay: 1s; }
        .float-item:nth-child(4) { bottom: 15%; right: 10%; animation-delay: 1.5s; }

        @keyframes float {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(5deg); }
        }

        /* Features Section */
        .features {
            padding: 6rem 5%;
            background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
        }

        .section-header {
            text-align: center;
            margin-bottom: 4rem;
        }

        .section-header h2 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }

        .section-header p {
            color: #a0aec0;
            font-size: 1.1rem;
        }

        .features-grid {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
        }

        .feature-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 2.5rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s;
            cursor: pointer;
        }

        .feature-card:hover {
            transform: translateY(-10px);
            background: rgba(255, 255, 255, 0.08);
            border-color: #667eea;
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        }

        .feature-icon {
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            margin-bottom: 1.5rem;
        }

        .feature-card h3 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }

        .feature-card p {
            color: #a0aec0;
            line-height: 1.6;
        }

        /* Workflow Section */
        .workflow {
            padding: 6rem 5%;
            background: #0a0e27;
        }

        .workflow-container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 4rem;
            margin-top: 3rem;
        }

        .workflow-item {
            position: relative;
            padding-left: 4rem;
        }

        .workflow-number {
            position: absolute;
            left: 0;
            top: 0;
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: 700;
        }

        .workflow-item h3 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }

        .workflow-item p {
            color: #a0aec0;
            line-height: 1.6;
        }

        /* CTA Section */
        .cta-section {
            padding: 6rem 5%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            text-align: center;
        }

        .cta-section h2 {
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }

        .cta-section p {
            font-size: 1.2rem;
            margin-bottom: 2rem;
            opacity: 0.9;
        }

        /* Footer */
        footer {
            padding: 3rem 5%;
            background: #0a0e27;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .footer-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 2rem;
        }

        @media (max-width: 768px) {
            .hero-content {
                grid-template-columns: 1fr;
                text-align: center;
            }

            .hero-text h1 {
                font-size: 2.5rem;
            }

            .nav-links { display: none; }
            .workflow-grid { grid-template-columns: 1fr; }
            .cube-container { width: 300px; height: 300px; }
        }
    </style>
</head>
<body>
    <!-- Main Content Wrapper -->
    <div id="main-content">
        <!-- Navigation -->
        <nav>
            <div class="nav-content">
                <div class="logo">Meta-Design</div>
                <ul class="nav-links">
                    <li><a href="#home">Home</a></li>
                    <li><a href="#features">Features</a></li>
                    <li><a href="#workflow">Workflow</a></li>
                </ul>
            </div>
        </nav>

        <!-- Hero Section -->
        <section class="hero" id="home">
            <div class="hero-content">
                <div class="hero-text">
                    <h1>Accelerate Materials Discovery with AI</h1>
                    <p>Meta-Design leverages advanced meta-learning models and physics-informed neural networks to optimize material formulations with unprecedented efficiency.</p>
                    <div class="cta-buttons">
                        <a href="/1_Data_Setup" target="_self" class="btn btn-primary">Start Discovery</a>
                        <a href="#workflow" class="btn btn-secondary">Learn More</a>
                    </div>
                </div>
                <div class="hero-visual">
                    <div class="cube-container">
                        <div class="floating-elements">
                            <div class="float-item">🧪</div>
                            <div class="float-item">🤖</div>
                            <div class="float-item">⚛️</div>
                            <div class="float-item">📈</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Features Section -->
        <section class="features" id="features">
            <div class="section-header">
                <h2>Advanced Features for Materials Science</h2>
                <p>Everything you need to revolutionize your materials discovery process</p>
            </div>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">🔬</div>
                    <h3>Digital Lab</h3>
                    <p>Automatically generate vast design spaces and new material formulations from your base components.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <h3>Meta-Learning Models</h3>
                    <p>Utilize MAML and Reptile to adapt quickly and make accurate predictions even with limited experimental data.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <h3>Advanced Optimization</h3>
                    <p>Employ Bayesian Optimization with multi-objective support (including Cost & CO2) to find the best trade-off solutions.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <h3>Physics-Informed AI</h3>
                    <p>Leverage PINNs that incorporate physical laws into the training process for more accurate and realistic predictions.</p>
                </div>
            </div>
        </section>

        <!-- Workflow Section -->
        <section class="workflow" id="workflow">
            <div class="workflow-container">
                <div class="section-header">
                    <h2>A Streamlined Workflow</h2>
                    <p>From data setup to AI-optimized discovery in a few simple steps</p>
                </div>
                <div class="workflow-grid">
                    <div class="workflow-item">
                        <div class="workflow-number">1</div>
                        <h3>Data Setup</h3>
                        <p>Upload your existing experimental data or use the Digital Lab to generate a new design space from scratch.</p>
                    </div>
                    <div class="workflow-item">
                        <div class="workflow-number">2</div>
                        <h3>Experimentation</h3>
                        <p>Select a model, configure its parameters, and run the experiment to get AI-driven suggestions for the next best experiments.</p>
                    </div>
                    <div class="workflow-item">
                        <div class="workflow-number">3</div>
                        <h3>Analysis & Iteration</h3>
                        <p>Analyze the results, visualize the trade-offs, and log new experimental findings to enrich the dataset for the next cycle.</p>
                    </div>
                    <div class="workflow-item">
                        <div class="workflow-number">4</div>
                        <h3>Discover</h3>
                        <p>Integrate lab results seamlessly and let the AI continuously refine its suggestions, accelerating your path to discovery.</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- CTA Section -->
        <section class="cta-section">
            <h2>Ready to Transform Your Materials Research?</h2>
            <p>Join leading researchers and engineers using Meta-Design for breakthrough discoveries.</p>
            <div class="cta-buttons" style="justify-content: center;">
                <a href="/1_Data_Setup" target="_self" class="btn btn-secondary">Get Started</a>
            </div>
        </section>

        <!-- Footer -->
        <footer>
            <div class="footer-content">
                <div style="color: #a0aec0;">
                    <div class="logo" style="margin-bottom: 0.5rem;">Meta-Design</div>
                </div>
            </div>
        </footer>
    </div>

    <script>
        // Smooth scroll for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    // Use a timer to ensure the element is available before scrolling
                    setTimeout(() => {
                        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }, 100);
                }
            });
        });

        // Add scroll animation
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -100px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);

        document.querySelectorAll('.feature-card, .workflow-item').forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(30px)';
            el.style.transition = 'all 0.6s ease-out';
            observer.observe(el);
        });
    </script>
</body>
</html>
"""

st.markdown(landing_page_html, unsafe_allow_html=True)
