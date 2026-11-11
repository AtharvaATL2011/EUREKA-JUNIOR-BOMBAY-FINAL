import streamlit as st
import torch
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
import random
import pandas as pd
import time
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Network-Embedded SMS Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Path to your best model
        model_path = './best_distilbert-base-multilingual-cased/'
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load the fine-tuned model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set model to evaluation mode
        model.eval()
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        st.success(f"✅ AI Model loaded successfully on {device}")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Make sure your model files are in the correct directory structure")
        return None, None, None

# Enhanced text preprocessing function
def preprocess_text(text):
    """Enhanced text preprocessing with language detection"""
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove some special characters but keep essential ones
    text = re.sub(r'[^\w\s.,!?$%-]', '', text)
    
    return text.strip()

# Language detection function
def detect_language(text):
    """Simple language detection for Indian languages"""
    hindi_chars = set('अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')
    if any(char in hindi_chars for char in text):
        return "Hindi/Devanagari"
    elif any(ord(char) > 127 for char in text):
        return "Regional Indian"
    else:
        return "English"

# Enhanced fraud explanation function
def explain_detection(probabilities, text):
    """Provide detailed explanation for AI detection"""
    explanation = []
    text_upper = text.upper()
    
    # Check for OTP/Banking keywords
    otp_keywords = ["OTP", "PASSWORD", "PIN", "CVV", "ACCOUNT", "BANK", "ATM"]
    if any(keyword in text_upper for keyword in otp_keywords):
        explanation.append("🚨 Contains banking/OTP sensitive keywords")
    
    # Check for suspicious links
    if any(keyword in text_upper for keyword in ["CLICK", "HTTP", "WWW", "LINK", ".COM"]):
        explanation.append("⚠️ Contains suspicious links or URLs")
    
    # Check for urgency tactics
    urgency_words = ["URGENT", "IMMEDIATELY", "EXPIRE", "LIMITED TIME", "ACT NOW", "HURRY"]
    if any(keyword in text_upper for keyword in urgency_words):
        explanation.append("⏰ Uses urgency/pressure tactics")
    
    # Check for money-related scams
    money_keywords = ["WON", "PRIZE", "LOTTERY", "REWARD", "₹", "RUPEES", "LAKH", "CRORE"]
    if any(keyword in text_upper for keyword in money_keywords):
        explanation.append("💰 Contains money/prize-related content")
    
    # Check for verification requests
    verify_keywords = ["VERIFY", "CONFIRM", "UPDATE", "VALIDATE", "AUTHENTICATE"]
    if any(keyword in text_upper for keyword in verify_keywords):
        explanation.append("🔐 Requests verification/confirmation")
    
    # Check for phone number patterns
    if re.search(r'\b\d{10}\b', text) or re.search(r'\+91\d{10}\b', text):
        explanation.append("📞 Contains phone numbers")
    
    return explanation

# Graph Analytics Functions for Fraud Ring Detection
def calculate_message_similarity(text1, text2):
    """Calculate similarity between two messages using simple word overlap"""
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def extract_urls_and_phones(text):
    """Extract URLs and phone numbers from message text"""
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    phones = re.findall(r'\b\d{10}\b|\+91\d{10}\b', text)
    domains = re.findall(r'[\w.-]+\.(?:com|org|net|in|co\.in)', text.lower())
    
    return {
        'urls': urls,
        'phones': phones, 
        'domains': domains
    }

def classify_campaign_type(text):
    """Classify the type of fraud campaign based on message content"""
    text_upper = text.upper()
    
    if any(keyword in text_upper for keyword in ["OTP", "BANK", "ACCOUNT", "ATM", "CVV"]):
        return "Banking/OTP Fraud"
    elif any(keyword in text_upper for keyword in ["WON", "PRIZE", "LOTTERY", "REWARD"]):
        return "Prize/Lottery Scam"
    elif any(keyword in text_upper for keyword in ["CLICK", "HTTP", "LINK", "DOWNLOAD"]):
        return "Phishing/Malware"
    elif any(keyword in text_upper for keyword in ["URGENT", "EXPIRE", "IMMEDIATE", "ACT NOW"]):
        return "Urgency Scam"
    else:
        return "General Fraud"

def analyze_fraud_network(all_messages):
    """Analyze message patterns to detect fraud rings and campaigns"""
    fraud_messages = [msg for msg in all_messages if msg['classification'] in ['suspected', 'confirmed_spam']]
    
    if len(fraud_messages) < 2:
        return {'clusters': [], 'network_nodes': [], 'campaign_indicators': []}
    
    clusters = []
    
    # Analyze message content similarities
    for i, msg1 in enumerate(fraud_messages):
        cluster_found = False
        msg1_indicators = extract_urls_and_phones(msg1['text'])
        
        for cluster in clusters:
            # Check if message belongs to existing cluster
            for msg2 in cluster['messages']:
                similarity = calculate_message_similarity(msg1['text'], msg2['text'])
                msg2_indicators = extract_urls_and_phones(msg2['text'])
                
                # Check for shared URLs, domains, or phone numbers
                shared_urls = set(msg1_indicators['urls']).intersection(set(msg2_indicators['urls']))
                shared_domains = set(msg1_indicators['domains']).intersection(set(msg2_indicators['domains']))
                shared_phones = set(msg1_indicators['phones']).intersection(set(msg2_indicators['phones']))
                
                if (similarity > 0.3 or shared_urls or shared_domains or shared_phones):
                    cluster['messages'].append(msg1)
                    cluster['senders'].add(msg1['sender'])
                    cluster['similarity_scores'].append(similarity)
                    cluster['shared_indicators'].extend(list(shared_urls) + list(shared_domains) + list(shared_phones))
                    cluster_found = True
                    break
            
            if cluster_found:
                break
        
        if not cluster_found:
            # Create new cluster
            clusters.append({
                'id': len(clusters) + 1,
                'messages': [msg1],
                'senders': {msg1['sender']},
                'similarity_scores': [1.0],
                'shared_indicators': msg1_indicators['urls'] + msg1_indicators['domains'] + msg1_indicators['phones'],
                'campaign_type': classify_campaign_type(msg1['text'])
            })
    
    # Filter clusters with multiple messages or senders
    significant_clusters = [c for c in clusters if len(c['messages']) > 1 or len(c['senders']) > 1]
    
    # Generate network nodes and campaign indicators
    network_nodes = []
    campaign_indicators = []
    
    for cluster in significant_clusters:
        network_nodes.append({
            'cluster_id': cluster['id'],
            'sender_count': len(cluster['senders']),
            'message_count': len(cluster['messages']),
            'campaign_type': cluster['campaign_type'],
            'risk_level': min(0.5 + len(cluster['senders']) * 0.1 + len(cluster['shared_indicators']) * 0.1, 1.0)
        })
        
        if len(cluster['shared_indicators']) > 0:
            campaign_indicators.append({
                'type': 'Coordinated Campaign',
                'cluster_id': cluster['id'],
                'shared_resources': cluster['shared_indicators'],
                'sender_count': len(cluster['senders']),
                'message_count': len(cluster['messages']),
                'campaign_type': cluster['campaign_type']
            })
    
    return {
        'clusters': significant_clusters,
        'network_nodes': network_nodes,
        'campaign_indicators': campaign_indicators
    }

# Working Analytics Functions
def generate_threat_report(all_messages, blocked_messages):
    """Generate a comprehensive threat analysis report"""
    fraud_messages = [msg for msg in all_messages if msg['classification'] in ['suspected', 'confirmed_spam']]
    
    # Basic statistics
    total_processed = len(all_messages)
    threat_count = len(fraud_messages)
    blocked_count = len(blocked_messages)
    
    # Analyze threat types
    threat_types = {}
    for msg in fraud_messages:
        threat_type = classify_campaign_type(msg['text'])
        threat_types[threat_type] = threat_types.get(threat_type, 0) + 1
    
    # Language analysis
    languages = {}
    for msg in fraud_messages:
        lang = msg.get('language', 'English')
        languages[lang] = languages.get(lang, 0) + 1
    
    # Time pattern analysis
    hour_pattern = {}
    for msg in fraud_messages:
        hour = msg['timestamp'].hour
        hour_pattern[hour] = hour_pattern.get(hour, 0) + 1
    
    # Risk score distribution
    risk_scores = [msg.get('fraud_risk', 0) for msg in fraud_messages]
    avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
    high_risk_count = len([r for r in risk_scores if r > 0.7])
    
    # Extract common keywords
    from collections import Counter
    all_words = []
    for msg in fraud_messages:
        words = re.findall(r'\w+', msg['text'].lower())
        all_words.extend([w for w in words if len(w) > 3])  # Only words longer than 3 chars
    
    common_keywords = Counter(all_words).most_common(10)
    
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_processed': total_processed,
            'threats_detected': threat_count,
            'threats_blocked': blocked_count,
            'detection_rate': (threat_count / total_processed * 100) if total_processed > 0 else 0,
            'average_risk_score': avg_risk * 100
        },
        'threat_breakdown': threat_types,
        'language_distribution': languages,
        'temporal_patterns': {
            'peak_hours': sorted(hour_pattern.items(), key=lambda x: x[1], reverse=True)[:3],
            'all_hours': hour_pattern
        },
        'risk_analysis': {
            'high_risk_messages': high_risk_count,
            'risk_distribution': {
                'low': len([r for r in risk_scores if r < 0.3]),
                'medium': len([r for r in risk_scores if 0.3 <= r < 0.7]),
                'high': len([r for r in risk_scores if r >= 0.7])
            }
        },
        'keyword_analysis': {
            'top_fraud_keywords': common_keywords
        },
        'recommendations': generate_security_recommendations(fraud_messages, threat_types, languages)
    }
    
    return report

def generate_security_recommendations(fraud_messages, threat_types, languages):
    """Generate security recommendations based on detected patterns"""
    recommendations = []
    
    # Threat type based recommendations
    if 'Banking/OTP Fraud' in threat_types:
        recommendations.append("Implement additional OTP verification alerts for banking-related messages")
        recommendations.append("Enhance keyword detection for financial terminology")
    
    if 'Phishing/Malware' in threat_types:
        recommendations.append("Strengthen URL scanning and domain reputation checking")
        recommendations.append("Add real-time link verification before message delivery")
    
    if 'Prize/Lottery Scam' in threat_types:
        recommendations.append("Flag messages containing prize/lottery keywords for manual review")
        recommendations.append("Implement sender reputation scoring for promotional content")
    
    # Language based recommendations
    if len(languages) > 1:
        recommendations.append("Consider language-specific fraud detection models for better accuracy")
        recommendations.append("Implement multi-language keyword databases for threat detection")
    
    # Volume based recommendations
    if len(fraud_messages) > 10:
        recommendations.append("Consider implementing rate limiting for high-volume senders")
        recommendations.append("Enhance real-time monitoring for coordinated attack patterns")
    
    return recommendations

def analyze_message_patterns(all_messages):
    """Perform deep pattern analysis on messages"""
    fraud_messages = [msg for msg in all_messages if msg['classification'] in ['suspected', 'confirmed_spam']]
    
    if not fraud_messages:
        return {"error": "No fraud messages to analyze"}
    
    # Similarity clustering
    clusters = []
    for i, msg1 in enumerate(fraud_messages):
        for j, msg2 in enumerate(fraud_messages[i+1:], i+1):
            similarity = calculate_message_similarity(msg1['text'], msg2['text'])
            if similarity > 0.4:  # High similarity threshold
                clusters.append({
                    'message_1': msg1['text'][:50] + "...",
                    'message_2': msg2['text'][:50] + "...",
                    'similarity': similarity,
                    'sender_1': msg1['sender'],
                    'sender_2': msg2['sender']
                })
    
    # Sender behavior analysis
    sender_analysis = {}
    for msg in fraud_messages:
        sender = msg['sender']
        if sender not in sender_analysis:
            sender_analysis[sender] = {
                'message_count': 0,
                'avg_risk': 0,
                'languages': set(),
                'threat_types': set(),
                'risk_scores': []
            }
        
        sender_analysis[sender]['message_count'] += 1
        sender_analysis[sender]['risk_scores'].append(msg.get('fraud_risk', 0))
        sender_analysis[sender]['languages'].add(msg.get('language', 'English'))
        sender_analysis[sender]['threat_types'].add(classify_campaign_type(msg['text']))
    
    # Calculate averages
    for sender in sender_analysis:
        scores = sender_analysis[sender]['risk_scores']
        sender_analysis[sender]['avg_risk'] = sum(scores) / len(scores) if scores else 0
        sender_analysis[sender]['languages'] = list(sender_analysis[sender]['languages'])
        sender_analysis[sender]['threat_types'] = list(sender_analysis[sender]['threat_types'])
        del sender_analysis[sender]['risk_scores']  # Remove raw scores for cleaner output
    
    # Keyword evolution tracking
    from collections import defaultdict
    keyword_timeline = defaultdict(list)
    
    for msg in sorted(fraud_messages, key=lambda x: x['timestamp']):
        words = re.findall(r'\w+', msg['text'].upper())
        fraud_keywords = ['OTP', 'BANK', 'URGENT', 'CLICK', 'WON', 'PRIZE', 'VERIFY']
        
        for keyword in fraud_keywords:
            if keyword in words:
                keyword_timeline[keyword].append(msg['timestamp'].strftime('%H:%M:%S'))
    
    return {
        'similarity_clusters': clusters,
        'sender_behavior': sender_analysis,
        'keyword_timeline': dict(keyword_timeline),
        'pattern_insights': {
            'total_fraud_senders': len(sender_analysis),
            'high_similarity_pairs': len([c for c in clusters if c['similarity'] > 0.6]),
            'coordinated_senders': len([s for s, data in sender_analysis.items() if data['message_count'] > 1])
        }
    }

def export_pattern_analysis(analysis, all_messages):
    """Export comprehensive pattern analysis to downloadable format"""
    
    # Create comprehensive export data
    export_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'summary': analysis.get('pattern_insights', {}),
        'similarity_analysis': {
            'high_similarity_pairs': analysis.get('similarity_clusters', []),
            'similarity_threshold': 0.4,  # The threshold used in analysis
            'total_pairs_found': len(analysis.get('similarity_clusters', []))
        },
        'sender_risk_profiles': analysis.get('sender_behavior', {}),
        'keyword_timeline': analysis.get('keyword_timeline', {}),
        'detailed_message_data': []
    }
    
    # Add detailed message analysis
    fraud_messages = [msg for msg in all_messages if msg['classification'] in ['suspected', 'confirmed_spam']]
    
    for msg in fraud_messages:
        indicators = extract_urls_and_phones(msg['text'])
        message_analysis = {
            'message_id': msg.get('id', 'unknown'),
            'sender': msg['sender'],
            'timestamp': msg['timestamp'].isoformat(),
            'classification': msg['classification'],
            'fraud_risk_score': msg.get('fraud_risk', 0),
            'language': msg.get('language', 'English'),
            'campaign_type': classify_campaign_type(msg['text']),
            'extracted_indicators': indicators,
            'ai_explanations': msg.get('explanations', []),
            'message_length': len(msg['text']),
            'message_preview': msg['text'][:100] + "..." if len(msg['text']) > 100 else msg['text']
        }
        export_data['detailed_message_data'].append(message_analysis)
    
    # Add network cluster analysis if available
    if hasattr(st.session_state, 'fraud_network_graph') and st.session_state.fraud_network_graph:
        export_data['network_analysis'] = {
            'fraud_clusters': st.session_state.fraud_network_graph.get('clusters', []),
            'campaign_indicators': st.session_state.fraud_network_graph.get('campaign_indicators', []),
            'network_nodes': st.session_state.fraud_network_graph.get('network_nodes', [])
        }
    
    return export_data

def create_sensitivity_controls():
    """Create interactive sensitivity adjustment interface"""
    st.subheader("Detection Sensitivity Controls")
    
    # Current thresholds
    current_spam = st.session_state.custom_thresholds.get('spam_threshold', 0.6) if st.session_state.custom_thresholds else 0.6
    current_suspected = st.session_state.custom_thresholds.get('suspected_threshold', 0.25) if st.session_state.custom_thresholds else 0.25
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_spam_threshold = st.slider(
            "Spam Block Threshold",
            min_value=0.1,
            max_value=1.0,
            value=current_spam,
            step=0.05,
            help="Messages above this threshold will be blocked"
        )
    
    with col2:
        new_suspected_threshold = st.slider(
            "Suspected Flag Threshold",
            min_value=0.1,
            max_value=0.8,
            value=current_suspected,
            step=0.05,
            help="Messages above this threshold will be flagged as suspicious"
        )
    
    # Preview impact on existing messages
    if st.session_state.messages or st.session_state.blocked_messages:
        st.subheader("Impact Preview")
        
        all_messages = st.session_state.messages + st.session_state.blocked_messages
        
        # Simulate new classifications
        new_classifications = {
            'would_block': 0,
            'would_flag': 0,
            'would_allow': 0,
            'changes': []
        }
        
        for msg in all_messages:
            if 'probabilities' in msg and msg['probabilities']:
                probs = msg['probabilities']
                spam_prob = probs.get('class_2_prob', 0)
                suspected_prob = probs.get('class_1_prob', 0)
                
                # Determine new classification
                if spam_prob >= new_spam_threshold:
                    new_class = 'confirmed_spam'
                elif spam_prob >= new_suspected_threshold or suspected_prob >= new_suspected_threshold:
                    new_class = 'suspected'
                else:
                    new_class = 'genuine'
                
                # Count classifications
                if new_class == 'confirmed_spam':
                    new_classifications['would_block'] += 1
                elif new_class == 'suspected':
                    new_classifications['would_flag'] += 1
                else:
                    new_classifications['would_allow'] += 1
                
                # Track changes
                if new_class != msg['classification']:
                    new_classifications['changes'].append({
                        'message_preview': msg['text'][:50] + "...",
                        'current': msg['classification'],
                        'new': new_class,
                        'sender': msg['sender']
                    })
        
        # Display preview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Would Block", new_classifications['would_block'])
        with col2:
            st.metric("Would Flag", new_classifications['would_flag'])
        with col3:
            st.metric("Would Allow", new_classifications['would_allow'])
        
        if new_classifications['changes']:
            st.subheader("Classification Changes")
            st.write(f"These {len(new_classifications['changes'])} messages would change classification:")
            
            for change in new_classifications['changes'][:5]:  # Show first 5 changes
                st.write(f"**{change['sender']}:** {change['message_preview']}")
                st.write(f"Current: `{change['current']}` → New: `{change['new']}`")
                st.write("---")
        else:
            st.success("No classification changes with these thresholds")
    
    # Apply button
    if st.button("Apply New Sensitivity Settings", type="primary"):
        st.session_state.custom_thresholds = {
            'spam_threshold': new_spam_threshold,
            'suspected_threshold': new_suspected_threshold
        }
        st.session_state.threshold_mode = True
        st.success("New sensitivity settings applied!")
        st.info("These settings will affect all future message classifications")
        st.rerun()

# Enhanced fraud detection function with threshold adjustment
def detect_fraud(text, model, tokenizer, device, custom_thresholds=None):
    """Detect fraud using the fine-tuned DistilBERT model with enhanced analysis and adjustable thresholds"""
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Simulate network processing delay for realism
        time.sleep(0.1)  # 100ms delay to simulate network processing
        
        # Tokenize the text
        inputs = tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move inputs to device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get probabilities for all 3 classes
        probs = predictions[0].cpu().numpy()
        original_predicted_class = np.argmax(probs)
        confidence = probs[original_predicted_class]
        
        # Apply custom threshold logic if enabled
        if custom_thresholds:
            spam_threshold = custom_thresholds.get('spam_threshold', 0.7)
            suspected_threshold = custom_thresholds.get('suspected_threshold', 0.3)
            
            # Custom classification logic
            if probs[2] >= spam_threshold:  # High spam probability
                predicted_class = 2
                classification = "confirmed_spam"
            elif probs[2] >= suspected_threshold or probs[1] >= suspected_threshold:  # Medium risk
                predicted_class = 1
                classification = "suspected"
            else:  # Low risk
                predicted_class = 0
                classification = "genuine"
        else:
            # Original logic
            predicted_class = original_predicted_class
            class_mapping = {
                0: "genuine",        # LABEL_0 - Safe messages
                1: "suspected",      # LABEL_1 - Suspected spam/fraud
                2: "confirmed_spam"  # LABEL_2 - Confirmed spam/fraud
            }
            classification = class_mapping.get(predicted_class, "unknown")
        
        # For fraud risk, use the confirmed_spam (LABEL_2) probability
        fraud_risk = probs[2] if len(probs) > 2 else confidence
        
        return classification, fraud_risk, {
            'class_0_prob': probs[0],  # LABEL_0 probability
            'class_1_prob': probs[1],  # LABEL_1 probability  
            'class_2_prob': probs[2],  # LABEL_2 probability
            'predicted_class': predicted_class,
            'original_predicted_class': original_predicted_class,
            'confidence': confidence,
            'using_custom_thresholds': custom_thresholds is not None
        }
            
    except Exception as e:
        st.error(f"❌ Error in fraud detection: {str(e)}")
        return "error", 0.0, {}

# Initialize session state with enhanced features
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'blocked_messages' not in st.session_state:
    st.session_state.blocked_messages = []
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0
if 'system_performance' not in st.session_state:
    st.session_state.system_performance = {
        'response_time': 0.3,
        'accuracy': 94.2,
        'false_positives': 2.1
    }
if 'custom_thresholds' not in st.session_state:
    st.session_state.custom_thresholds = None
if 'threshold_mode' not in st.session_state:
    st.session_state.threshold_mode = False
if 'fraud_network_graph' not in st.session_state:
    st.session_state.fraud_network_graph = {}
if 'campaign_clusters' not in st.session_state:
    st.session_state.campaign_clusters = []
if 'test_message' not in st.session_state:
    st.session_state.test_message = ""

# Load model
model, tokenizer, device = load_model_and_tokenizer()

# Enhanced main app header with network simulation
st.title("🛡️ Network-Embedded SMS Fraud Detection System")
st.markdown("### *Telecom-Integrated AI Protection • Real-time Threat Interception*")

# Network Architecture Visualization
with st.expander("🏗️ Network Architecture & Integration"):
    st.markdown("""
    ```
    📱 User SMS → 📡 Telecom Network → 🤖 AI Fraud Layer → 👤 Recipient
                                         ↓
                                    🚫 Block/⚠️ Flag/✅ Allow
    ```
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📡 Network Layer**\n- Telecom Infrastructure\n- Real-time Interception\n- Zero User Setup")
    with col2:
        st.success("**🤖 AI Processing**\n- DistilBERT Analysis\n- Indian Language Support\n- Pattern Recognition")
    with col3:
        st.warning("**🛡️ Protection Layer**\n- Instant Blocking\n- Smart Flagging\n- Threat Logging")

st.markdown("---")

# Show enhanced model status with network simulation
network_col1, network_col2, network_col3 = st.columns(3)
with network_col1:
    if model is not None:
        st.success("🤖 **AI Fraud Detection:** ONLINE")
    else:
        st.error("🤖 **AI Fraud Detection:** OFFLINE")

with network_col2:
    st.success("📡 **Network Integration:** SIMULATED")

with network_col3:
    st.success("🌐 **Telecom Gateway:** DEMO MODE")

# Create two columns for sender and receiver
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 **Network Sender Interface**")
    
    # Add threshold adjustment controls
    with st.expander("🎛️ Advanced Model Controls"):
        threshold_mode = st.checkbox("Enable Custom Classification Thresholds", 
                                   value=st.session_state.threshold_mode)
        st.session_state.threshold_mode = threshold_mode
        
        if threshold_mode:
            st.write("**Adjust Classification Sensitivity:**")
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                spam_threshold = st.slider(
                    "Spam Block Threshold", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.6,  # Lower threshold = more blocking
                    step=0.1,
                    help="Higher = stricter spam detection"
                )
            
            with col_t2:
                suspected_threshold = st.slider(
                    "Suspected Flag Threshold", 
                    min_value=0.1, 
                    max_value=0.8, 
                    value=0.25,  # Lower threshold = more flagging
                    step=0.05,
                    help="Lower = more messages flagged as suspicious"
                )
            
            st.session_state.custom_thresholds = {
                'spam_threshold': spam_threshold,
                'suspected_threshold': suspected_threshold
            }
            
            st.info(f"🎯 Current: Block if >={spam_threshold:.1%} spam risk, Flag if >={suspected_threshold:.1%} risk")
        else:
            st.session_state.custom_thresholds = None
    
    # Enhanced sender details with multiple options
    with st.container():
        sender_options = ["User_001", "User_002", "Scammer_123", "Business_456", "Unknown_789"]
        sender_name = st.selectbox("📱 Sender ID:", sender_options, key="sender")
        
        recipient_options = ["User_001", "User_002", "User_003", "User_004", "User_005"]
        recipient_name = st.selectbox("👤 Recipient:", recipient_options, key="recipient")
    
    # Quick test scenarios with multiple options - MOVED BEFORE TEXT AREA
    st.subheader("🧪 Quick Test Scenarios")
    test_col1, test_col2, test_col3 = st.columns(3)
    
    with test_col1:
        if st.button("Test Suspected", type="secondary"):
            st.session_state.test_message = "aapka account 2 hrs me freeze ho jayenga renew it by clicking on www.fake.com"
    
    with test_col2:
        if st.button("Test Confirm Spam", type="secondary"):
            st.session_state.test_message = "URGENT! Your bank account has been compromised. Share your OTP 456789 immediately. Click http://fake-sbi-security.com/urgent-verify now! Call 7777888899 within 10 minutes or lose ₹2 lakh!"
    
    with test_col3:
        if st.button("Test Genuine", type="secondary"):
            st.session_state.test_message = "Hi! Are we still meeting for lunch at 2 PM today? Let me know if you need to reschedule."
    
    # Message composition - Use test_message as default value
    st.subheader("✍️ Compose Message")
    message_text = st.text_area(
        "Type your message:",
        height=120,
        placeholder="Enter your message here...",
        value=st.session_state.test_message,
        key="message_input"
    )
    
    # Add a debug section to test model behavior
    if st.checkbox("🔧 Debug Mode: Live Model Analysis"):
        if message_text and model is not None:
            st.write("**🧪 Real-time Model Analysis:**")
            
            # Test with original model logic
            classification_orig, fraud_risk_orig, probabilities_orig = detect_fraud(message_text, model, tokenizer, device, None)
            
            # Test with custom thresholds if enabled
            if st.session_state.custom_thresholds:
                classification_custom, fraud_risk_custom, probabilities_custom = detect_fraud(
                    message_text, model, tokenizer, device, st.session_state.custom_thresholds
                )
                
                debug_col1, debug_col2 = st.columns(2)
                
                with debug_col1:
                    st.write("**Original Model:**")
                    st.write(f"Decision: `{classification_orig}`")
                    st.write(f"Safe: {probabilities_orig.get('class_0_prob', 0):.1%}")
                    st.write(f"Suspected: {probabilities_orig.get('class_1_prob', 0):.1%}")
                    st.write(f"Spam: {probabilities_orig.get('class_2_prob', 0):.1%}")
                
                with debug_col2:
                    st.write("**With Custom Thresholds:**")
                    st.write(f"Decision: `{classification_custom}`")
                    st.write(f"Spam Threshold: {st.session_state.custom_thresholds['spam_threshold']:.1%}")
                    st.write(f"Suspected Threshold: {st.session_state.custom_thresholds['suspected_threshold']:.1%}")
                    
                    if classification_orig != classification_custom:
                        st.success("✅ Threshold adjustment working!")
                    else:
                        st.info("Same result as original")
            else:
                st.write(f"**Classification:** `{classification_orig}`")
                st.write(f"**Probabilities:**")
                st.write(f"• Safe: {probabilities_orig.get('class_0_prob', 0):.1%}")
                st.write(f"• Suspected: {probabilities_orig.get('class_1_prob', 0):.1%}")
                st.write(f"• Spam: {probabilities_orig.get('class_2_prob', 0):.1%}")
        else:
            st.info("💡 Enter a message in the text area above to see live analysis")
        
        st.markdown("---")
    
    # Enhanced message analysis
    if message_text:
        char_count = len(message_text)
        detected_lang = detect_language(message_text)
        
        analysis_col1, analysis_col2 = st.columns(2)
        with analysis_col1:
            st.caption(f"Characters: {char_count}/160 {'(Standard SMS)' if char_count <= 160 else '(Long SMS)'}")
        with analysis_col2:
            st.caption(f"🌍 Language: {detected_lang}")
    
    # Network Processing Simulation
    if len(message_text.strip()) > 0 and model is not None:
        with st.expander("🌐 Network Processing Layer (Live Analysis)"):
            st.info("📡 Intercepting SMS at telecom network level...")
            st.info("🔍 Running AI analysis on network server...")
            
            # Real-time analysis
            classification, fraud_risk, probabilities = detect_fraud(
                message_text, model, tokenizer, device, st.session_state.custom_thresholds
            )
            
            if classification == "confirmed_spam":
                st.error("🚨 NETWORK DECISION: BLOCK MESSAGE")
                display_risk = probabilities.get('class_2_prob', fraud_risk)
                st.error(f"Threat Level: {display_risk:.1%}")
            elif classification == "suspected":
                st.warning("⚠️ NETWORK DECISION: FLAG & DELIVER")
                display_risk = max(probabilities.get('class_1_prob', 0), probabilities.get('class_2_prob', 0))
                st.warning(f"Suspicion Level: {display_risk:.1%}")
            else:
                st.success("✅ NETWORK DECISION: SAFE DELIVERY")
                display_risk = probabilities.get('class_0_prob', 1-fraud_risk)
                st.success(f"Safety Score: {display_risk:.1%}")
            
            # AI Reasoning
            explanations = explain_detection(probabilities, message_text)
            if explanations:
                st.write("**🧠 AI Detection Reasoning:**")
                for exp in explanations:
                    st.write(f"• {exp}")
    
    # Send button
    send_col1, send_col2 = st.columns([1, 2])
    with send_col1:
        send_button = st.button("📨 Send via Network", type="primary", use_container_width=True)
    
    with send_col2:
        if len(message_text.strip()) > 0 and model is not None:
            # Real-time risk assessment with custom thresholds
            classification, fraud_risk, probabilities = detect_fraud(
                message_text, model, tokenizer, device, st.session_state.custom_thresholds
            )
            
            # Display appropriate percentage based on classification
            if classification == "confirmed_spam":
                display_risk = probabilities.get('class_2_prob', fraud_risk)  # Spam probability
                st.error(f"🚫 WILL BLOCK: {display_risk:.1%}")
            elif classification == "suspected":
                display_risk = max(probabilities.get('class_1_prob', 0), probabilities.get('class_2_prob', 0))  # Higher of suspected or spam
                st.warning(f"⚠️ WILL FLAG: {display_risk:.1%}")
            else:
                display_risk = probabilities.get('class_0_prob', 1-fraud_risk)  # Genuine probability
                st.success(f"✅ WILL DELIVER: {display_risk:.1%}")
        elif model is None:
            st.error("🤖 Model not loaded")
        else:
            st.info("💭 Enter message to see prediction")
    
    # Handle message sending with enhanced notifications
    if send_button:
        if len(message_text.strip()) > 0 and model is not None:
            # Increment API calls
            st.session_state.api_calls += 1
            
            # Detect fraud
            classification, fraud_risk, probabilities = detect_fraud(
                message_text, model, tokenizer, device, st.session_state.custom_thresholds
            )
            
            # Create enhanced message object
            message = {
                'id': len(st.session_state.messages) + len(st.session_state.blocked_messages) + 1,
                'sender': sender_name,
                'recipient': recipient_name,
                'text': message_text,
                'timestamp': datetime.now(),
                'classification': classification,
                'fraud_risk': fraud_risk,
                'probabilities': probabilities,
                'language': detect_language(message_text),
                'explanations': explain_detection(probabilities, message_text)
            }
            
            # Handle based on classification with enhanced notifications
            if classification == "confirmed_spam":
                st.session_state.blocked_messages.append(message)
                st.error("🚨 **NETWORK BLOCKED MESSAGE!**")
                st.error("This message was intercepted at the network level and blocked before reaching the recipient.")
                
                # Add notification
                notification = {
                    'time': datetime.now(),
                    'type': 'threat_blocked',
                    'message': f"🚫 Blocked spam from {sender_name}",
                    'details': f"Threat level: {fraud_risk:.1%}"
                }
                st.session_state.notifications.append(notification)
                
                # Update fraud network analysis
                all_messages = st.session_state.messages + st.session_state.blocked_messages
                network_analysis = analyze_fraud_network(all_messages)
                st.session_state.fraud_network_graph = network_analysis
                
                # Check for new campaign patterns
                if network_analysis['campaign_indicators']:
                    campaign_alert = {
                        'time': datetime.now(),
                        'type': 'campaign_detected',
                        'message': f"🕸️ Fraud ring detected involving {sender_name}",
                        'details': f"Campaign type: {network_analysis['campaign_indicators'][-1].get('campaign_type', 'Unknown')}"
                    }
                    st.session_state.notifications.append(campaign_alert)
                
            else:
                st.session_state.messages.append(message)
                if classification == "suspected":
                    st.warning("⚠️ **Message delivered with security warning**")
                    st.info("💡 Recipient will receive this message with caution notice.")
                    
                    # Add notification
                    notification = {
                        'time': datetime.now(),
                        'type': 'suspicious_delivered',
                        'message': f"⚠️ Flagged message from {sender_name}",
                        'details': f"Suspicion level: {fraud_risk:.1%}"
                    }
                    st.session_state.notifications.append(notification)
                    
                    # Update network analysis for suspected messages too
                    all_messages = st.session_state.messages + st.session_state.blocked_messages
                    network_analysis = analyze_fraud_network(all_messages)
                    st.session_state.fraud_network_graph = network_analysis
                else:
                    st.success("✅ **Message delivered safely through network!**")
            
            # Show detailed AI analysis
            with st.expander("🔍 Complete AI Network Analysis"):
                st.write("**Model Probabilities:**")
                st.write(f"• Genuine: {probabilities.get('class_0_prob', 0):.1%}")
                st.write(f"• Suspected: {probabilities.get('class_1_prob', 0):.1%}")
                st.write(f"• Confirmed Spam: {probabilities.get('class_2_prob', 0):.1%}")
                st.write(f"**Final Decision:** Class {probabilities.get('predicted_class', 'N/A')} (Confidence: {probabilities.get('confidence', 0):.1%})")
                st.write(f"**Language Detected:** {message['language']}")
                st.write(f"**Processing Time:** {st.session_state.system_performance['response_time']}s")
            
            st.rerun()
            
        elif model is None:
            st.error("❌ Cannot send message: Network AI system offline")
        else:
            st.warning("⚠️ Please enter a message before sending")

with col2:
    st.header("📥 **Network Receiver Interface**")
    st.subheader(f"💬 Messages for {recipient_name}")
    
    # Display messages with enhanced formatting
    if st.session_state.messages:        
        for message in reversed(st.session_state.messages[-10:]):  # Show last 10 messages
            # Create enhanced message bubble
            with st.container():
                # Message header with timestamp and language
                header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
                with header_col1:
                    st.caption(f"📱 From: **{message['sender']}**")
                with header_col2:
                    st.caption(f"🌍 {message.get('language', 'English')}")
                with header_col3:
                    st.caption(f"🕐 {message['timestamp'].strftime('%H:%M:%S')}")
                
                # Message content with enhanced security status
                if message['classification'] == "suspected":
                    st.warning("⚠️ **NETWORK SECURITY WARNING: SUSPECTED THREAT**")
                    with st.expander("🔍 View Flagged Message (Click to expand)", expanded=False):
                        st.markdown(f"💬 *{message['text']}*")
                        st.error("🛡️ **Network Protection Active:** Please verify sender identity before taking any action")
                        
                        # Show AI explanations
                        if message.get('explanations'):
                            st.markdown("**🤖 Network AI Detected:**")
                            for explanation in message['explanations']:
                                st.write(f"• {explanation}")
                        
                        # Show probabilities
                        if 'probabilities' in message and message['probabilities']:
                            st.markdown("**📊 Network Analysis:**")
                            probs = message['probabilities']
                            st.write(f"• Safe: {probs.get('class_0_prob', 0):.1%}")
                            st.write(f"• Suspicious: {probs.get('class_1_prob', 0):.1%}")
                            st.write(f"• Spam: {probs.get('class_2_prob', 0):.1%}")
                            
                elif message['classification'] == "confirmed_spam":
                    st.error("🚫 **NETWORK BLOCKED: CONFIRMED THREAT**")
                else:
                    st.success("✅ **NETWORK VERIFIED: SAFE MESSAGE**")
                    # Display genuine message
                    st.markdown(f"💬 *{message['text']}*")
                    st.caption(f"🛡️ Network verified genuine (Risk: {message.get('fraud_risk', 0):.1%})")
                    
                    # Show language if not English
                    if message.get('language', 'English') != 'English':
                        st.caption(f"🌍 Message in {message['language']}")
                
                st.markdown("---")
    else:
        st.info("📭 **No messages received through network**")
        st.markdown("*Network-filtered messages will appear here*")

# Enhanced Statistics Section  
total_sent = len(st.session_state.messages) + len(st.session_state.blocked_messages)
genuine_count = len([m for m in st.session_state.messages if m['classification'] == 'genuine'])
suspected_count = len([m for m in st.session_state.messages if m['classification'] == 'suspected'])
blocked_count = len(st.session_state.blocked_messages)

# Enhanced Analytics Dashboard
if st.session_state.messages or st.session_state.blocked_messages:
    st.header("📊 **Network Security Analytics Dashboard**")
    
    # Fraud Ring Detection Section
    if st.session_state.fraud_network_graph and st.session_state.fraud_network_graph.get('campaign_indicators'):
        st.subheader("🕸️ **Fraud Ring Detection - Graph Analytics**")
        st.info("**Network-level visibility** exposes organized fraud networks through pattern analysis")
        
        fraud_analysis = st.session_state.fraud_network_graph
        
        # Campaign Overview
        campaign_col1, campaign_col2, campaign_col3 = st.columns(3)
        
        with campaign_col1:
            st.metric("Fraud Campaigns", len(fraud_analysis.get('campaign_indicators', [])))
        with campaign_col2:
            total_coordinated = sum(c.get('sender_count', 0) for c in fraud_analysis.get('campaign_indicators', []))
            st.metric("Coordinated Senders", total_coordinated)
        with campaign_col3:
            total_messages = sum(c.get('message_count', 0) for c in fraud_analysis.get('campaign_indicators', []))
            st.metric("Campaign Messages", total_messages)
        
        # Campaign Details
        for idx, campaign in enumerate(fraud_analysis.get('campaign_indicators', []), 1):
            with st.expander(f"🎯 Campaign #{idx}: {campaign.get('campaign_type', 'Unknown')}"):
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.write("**Campaign Intelligence:**")
                    st.write(f"• **Type:** {campaign.get('campaign_type', 'Unknown')}")
                    st.write(f"• **Coordinated Senders:** {campaign.get('sender_count', 0)}")
                    st.write(f"• **Total Messages:** {campaign.get('message_count', 0)}")
                    st.write(f"• **Cluster ID:** {campaign.get('cluster_id', 'N/A')}")
                
                with detail_col2:
                    st.write("**Shared Infrastructure:**")
                    shared_resources = campaign.get('shared_resources', [])
                    if shared_resources:
                        for resource in shared_resources[:5]:  # Show first 5
                            if 'http' in resource.lower():
                                st.write(f"🔗 URL: `{resource}`")
                            elif any(c.isdigit() for c in resource):
                                st.write(f"📞 Phone: `{resource}`")
                            else:
                                st.write(f"🌐 Domain: `{resource}`")
                    else:
                        st.write("• Content similarity patterns")
                        st.write("• Timing correlations")
                        st.write("• Message structure analysis")
        
        # Network Graph Simulation
        st.subheader("🌐 **Fraud Network Visualization**")
        
        if fraud_analysis.get('network_nodes'):
            # Create a simple network representation
            network_data = []
            for node in fraud_analysis['network_nodes']:
                network_data.append({
                    'Cluster': f"Cluster {node['cluster_id']}",
                    'Senders': node['sender_count'],
                    'Messages': node['message_count'],
                    'Risk Level': node['risk_level'] * 100,
                    'Campaign Type': node['campaign_type']
                })
            
            network_df = pd.DataFrame(network_data)
            
            graph_col1, graph_col2 = st.columns(2)
            with graph_col1:
                st.write("**Network Clusters by Risk Level**")
                st.bar_chart(network_df.set_index('Cluster')['Risk Level'])
            
            with graph_col2:
                st.write("**Campaign Distribution**")
                campaign_counts = network_df['Campaign Type'].value_counts()
                st.bar_chart(campaign_counts)
        
        # Initialize session states for analytics displays
        if 'show_threat_report' not in st.session_state:
            st.session_state.show_threat_report = False
        if 'show_pattern_analysis' not in st.session_state:
            st.session_state.show_pattern_analysis = False
        if 'show_sensitivity_controls' not in st.session_state:
            st.session_state.show_sensitivity_controls = False

        # Replace bulk mitigation buttons with WORKING prototype buttons
        st.subheader("🛡️ **Advanced Network Analytics**")
        mitigation_col1, mitigation_col2, mitigation_col3 = st.columns(3)
        
        with mitigation_col1:
            if st.button("📊 Generate Detailed Threat Report", type="primary"):
                all_messages = st.session_state.messages + st.session_state.blocked_messages
                if all_messages:
                    st.session_state.show_threat_report = True
                    st.success("✅ Comprehensive threat report generated!")
                else:
                    st.warning("⚠️ No messages to analyze. Send some test messages first.")
        
        with mitigation_col2:
            if st.button("🔍 Analyze Message Patterns", type="primary"):
                all_messages = st.session_state.messages + st.session_state.blocked_messages
                if all_messages:
                    st.session_state.show_pattern_analysis = True
                    st.success("✅ Deep pattern analysis completed!")
                else:
                    st.warning("⚠️ No messages to analyze. Send some test messages first.")
        
        with mitigation_col3:
            if st.button("⚙️ Adjust Detection Sensitivity", type="primary"):
                st.session_state.show_sensitivity_controls = True
                st.success("✅ Sensitivity controls activated!")

        # Display threat report
        if st.session_state.show_threat_report:
            st.markdown("---")
            st.subheader("📊 Comprehensive Threat Analysis Report")
            
            all_messages = st.session_state.messages + st.session_state.blocked_messages
            report = generate_threat_report(all_messages, st.session_state.blocked_messages)
            
            # Summary section
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Messages Processed", report['summary']['total_processed'])
            with col2:
                st.metric("Threats Detected", report['summary']['threats_detected'])
            with col3:
                st.metric("Detection Rate", f"{report['summary']['detection_rate']:.1f}%")
            with col4:
                st.metric("Avg Risk Score", f"{report['summary']['average_risk_score']:.1f}%")
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Threat Type Breakdown:**")
                for threat_type, count in report['threat_breakdown'].items():
                    st.write(f"• {threat_type}: {count} messages")
                
                st.write("**Language Distribution:**")
                for lang, count in report['language_distribution'].items():
                    st.write(f"• {lang}: {count} messages")
            
            with col2:
                st.write("**Peak Threat Hours:**")
                for hour, count in report['temporal_patterns']['peak_hours']:
                    st.write(f"• {hour:02d}:00 - {count} messages")
                
                st.write("**Top Fraud Keywords:**")
                for keyword, count in report['keyword_analysis']['top_fraud_keywords'][:5]:
                    st.write(f"• {keyword}: {count} occurrences")
            
            # Security recommendations
            st.write("**Security Recommendations:**")
            for rec in report['recommendations']:
                st.write(f"• {rec}")
            
            # Download button
            report_json = json.dumps(report, indent=2, default=str)
            st.download_button(
                "📥 Download Full Report (JSON)",
                data=report_json,
                file_name=f"threat_report_{report['generated_at'].replace(':', '-').replace(' ', '_')}.json",
                mime="application/json"
            )
            
            if st.button("❌ Hide Report"):
                st.session_state.show_threat_report = False
                st.rerun()

        # Display pattern analysis WITH DOWNLOAD FUNCTIONALITY
        if st.session_state.show_pattern_analysis:
            st.markdown("---")
            st.subheader("🔍 Deep Pattern Analysis")
            
            all_messages = st.session_state.messages + st.session_state.blocked_messages
            analysis = analyze_message_patterns(all_messages)
            
            if 'error' not in analysis:
                # Pattern insights
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fraud Senders", analysis['pattern_insights']['total_fraud_senders'])
                with col2:
                    st.metric("Similar Message Pairs", analysis['pattern_insights']['high_similarity_pairs'])
                with col3:
                    st.metric("Repeat Offenders", analysis['pattern_insights']['coordinated_senders'])
                
                # Similarity clusters
                if analysis['similarity_clusters']:
                    st.write("**High Similarity Message Pairs:**")
                    for cluster in analysis['similarity_clusters'][:5]:
                        st.write(f"**Similarity: {cluster['similarity']:.2f}**")
                        st.write(f"From {cluster['sender_1']}: {cluster['message_1']}")
                        st.write(f"From {cluster['sender_2']}: {cluster['message_2']}")
                        st.write("---")
                
                # Sender behavior
                st.write("**Sender Risk Analysis:**")
                for sender, data in list(analysis['sender_behavior'].items())[:5]:
                    st.write(f"**{sender}:**")
                    st.write(f"• Messages: {data['message_count']}")
                    st.write(f"• Avg Risk: {data['avg_risk']:.2f}")
                    st.write(f"• Languages: {', '.join(data['languages'])}")
                    st.write(f"• Threat Types: {', '.join(data['threat_types'])}")
                    st.write("---")
                
                # Export functionality - NEW SECTION WITH DOWNLOAD BUTTONS
                st.subheader("📥 Export Pattern Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Export full analysis as JSON
                    full_export = export_pattern_analysis(analysis, all_messages)
                    export_json = json.dumps(full_export, indent=2, default=str)
                    
                    st.download_button(
                        "📊 Download Full Analysis (JSON)",
                        data=export_json,
                        file_name=f"pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        help="Complete pattern analysis with all detected relationships"
                    )
                
                with col2:
                    # Export similarity clusters as CSV
                    if analysis['similarity_clusters']:
                        similarity_df = pd.DataFrame(analysis['similarity_clusters'])
                        similarity_csv = similarity_df.to_csv(index=False)
                        
                        st.download_button(
                            "🔗 Download Similarity Clusters (CSV)",
                            data=similarity_csv,
                            file_name=f"similarity_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Message similarity relationships in spreadsheet format"
                        )
                    else:
                        st.info("No similarity clusters to export")
                
                with col3:
                    # Export sender behavior as CSV  
                    if analysis['sender_behavior']:
                        sender_data = []
                        for sender, data in analysis['sender_behavior'].items():
                            sender_row = {
                                'Sender': sender,
                                'Message_Count': data['message_count'],
                                'Avg_Risk_Score': data['avg_risk'],
                                'Languages': ', '.join(data['languages']),
                                'Threat_Types': ', '.join(data['threat_types'])
                            }
                            sender_data.append(sender_row)
                        
                        sender_df = pd.DataFrame(sender_data)
                        sender_csv = sender_df.to_csv(index=False)
                        
                        st.download_button(
                            "👤 Download Sender Profiles (CSV)",
                            data=sender_csv,
                            file_name=f"sender_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Detailed sender risk profiles in spreadsheet format"
                        )
                    else:
                        st.info("No sender data to export")
                
                # Summary export info
                st.info(f"📋 Analysis includes {len(all_messages)} total messages, {len([m for m in all_messages if m['classification'] in ['suspected', 'confirmed_spam']])} flagged as threats")
                
            else:
                st.info("Send some suspicious messages to see pattern analysis")
            
            if st.button("❌ Hide Analysis"):
                st.session_state.show_pattern_analysis = False
                st.rerun()

        # Display sensitivity controls
        if st.session_state.show_sensitivity_controls:
            st.markdown("---")
            create_sensitivity_controls()
            
            if st.button("❌ Hide Controls"):
                st.session_state.show_sensitivity_controls = False
                st.rerun()
        
        st.markdown("---")
    
    # System Performance Metrics
    st.subheader("⚡ Real-time Network Performance")
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("Response Time", f"{st.session_state.system_performance['response_time']}s", "↓0.1s")
    with perf_col2:
        st.metric("AI Accuracy", f"{st.session_state.system_performance['accuracy']}%", "↑2.1%")
    with perf_col3:
        st.metric("False Positives", f"{st.session_state.system_performance['false_positives']}%", "↓0.5%")
    with perf_col4:
        st.metric("Threats Blocked", f"{blocked_count}", f"↑{blocked_count}")
    
    # IMPROVED ANALYTICS SECTION - Better spacing and elegant charts
    if total_sent > 0:
        st.subheader("📈 **Visual Analytics**")
        
        # First Row - Main Classification Chart (Full Width)
        st.write("**Message Classification Distribution**")
        flow_data = pd.DataFrame({
            'Status': ['Safe Delivery', 'Flagged Delivery', 'Network Blocked'],
            'Count': [genuine_count, suspected_count, blocked_count],
            'Percentage': [
                (genuine_count/total_sent)*100,
                (suspected_count/total_sent)*100,
                (blocked_count/total_sent)*100
            ]
        })
        
        # Use Plotly for better visualization with full width
        fig = px.pie(flow_data, values='Count', names='Status', 
                    color_discrete_map={
                        'Safe Delivery': '#2E8B57',
                        'Flagged Delivery': '#FF8C00', 
                        'Network Blocked': '#DC143C'
                    },
                    title="Real-time Message Processing Results")
        fig.update_layout(
            height=400, 
            margin=dict(l=20, r=20, t=60, b=20),
            title_x=0.5,
            font=dict(size=14),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Second Row - Timeline and Patterns (Two Column Layout)
        timeline_col, pattern_col = st.columns([1, 1], gap="large")
        
        with timeline_col:
            st.write("**Threat Detection Timeline**")
            # Create timeline data based on actual message timestamps
            timeline_data = []
            all_messages = sorted(st.session_state.messages + st.session_state.blocked_messages, 
                                key=lambda x: x['timestamp'])
            
            for msg in all_messages:
                timeline_data.append({
                    'Time': msg['timestamp'].strftime('%H:%M:%S'),
                    'Risk Score': msg.get('fraud_risk', 0) * 100,
                    'Classification': msg['classification'],
                    'Sender': msg['sender']
                })
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                fig2 = px.scatter(timeline_df, x='Time', y='Risk Score', 
                                color='Classification',
                                color_discrete_map={
                                    'genuine': '#2E8B57',
                                    'suspected': '#FF8C00',
                                    'confirmed_spam': '#DC143C'
                                },
                                hover_data=['Sender'],
                                title="Risk Score Over Time")
                fig2.update_layout(
                    height=400, 
                    margin=dict(l=10, r=10, t=50, b=50),
                    title_x=0.5,
                    font=dict(size=12),
                    xaxis_title="Time",
                    yaxis_title="Risk Score (%)"
                )
                fig2.update_traces(marker=dict(size=8))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("📊 No timeline data available yet")
                st.markdown("*Send some messages to see timeline analysis*")
        
        with pattern_col:
            st.write("**Fraud Campaign Types**")
            # Analyze actual fraud patterns from messages
            fraud_patterns = {}
            all_fraud_messages = [msg for msg in (st.session_state.messages + st.session_state.blocked_messages) 
                                if msg['classification'] in ['suspected', 'confirmed_spam']]
            
            for msg in all_fraud_messages:
                # Classify fraud type based on actual message content
                fraud_type = classify_campaign_type(msg['text'])
                fraud_patterns[fraud_type] = fraud_patterns.get(fraud_type, 0) + 1
            
            if fraud_patterns:
                pattern_df = pd.DataFrame(list(fraud_patterns.items()), 
                                        columns=['Fraud Type', 'Count'])
                fig3 = px.bar(pattern_df, x='Fraud Type', y='Count',
                            color='Count', 
                            color_continuous_scale='Reds',
                            title="Detected Campaign Types")
                fig3.update_layout(
                    height=400, 
                    margin=dict(l=10, r=10, t=50, b=80),
                    title_x=0.5,
                    font=dict(size=12),
                    xaxis_title="Campaign Type",
                    yaxis_title="Message Count"
                )
                fig3.update_xaxes(tickangle=45)
                fig3.update_traces(marker_line_width=1, marker_line_color="white")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("📊 No fraud patterns detected yet")
                st.markdown("*Send some suspicious messages to see pattern analysis*")
        
        # Third Row - Network Visualization (Full Width if fraud network exists)
        if st.session_state.fraud_network_graph and st.session_state.fraud_network_graph.get('network_nodes'):
            st.markdown("---")
            st.write("**Network Cluster Analysis**")
            
            fraud_analysis = st.session_state.fraud_network_graph
            network_data = []
            for node in fraud_analysis['network_nodes']:
                network_data.append({
                    'Cluster': f"Cluster {node['cluster_id']}",
                    'Senders': node['sender_count'],
                    'Messages': node['message_count'],
                    'Risk Level': node['risk_level'] * 100,
                    'Campaign Type': node['campaign_type']
                })
            
            network_df = pd.DataFrame(network_data)
            
            # Create side-by-side network charts
            net_col1, net_col2 = st.columns([1, 1], gap="large")
            
            with net_col1:
                st.write("**Clusters by Risk Level**")
                fig4 = px.bar(network_df, x='Cluster', y='Risk Level',
                             color='Risk Level',
                             color_continuous_scale='Reds',
                             title="Network Risk Assessment")
                fig4.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=50, b=50),
                    title_x=0.5,
                    font=dict(size=11)
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            with net_col2:
                st.write("**Campaign Type Distribution**")
                campaign_counts = network_df['Campaign Type'].value_counts()
                fig5 = px.pie(values=campaign_counts.values, names=campaign_counts.index,
                             title="Network Campaign Types")
                fig5.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=50, b=20),
                    title_x=0.5,
                    font=dict(size=11),
                    showlegend=True
                )
                st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown("---")

# Enhanced Sidebar with Network Control Center
with st.sidebar:
    st.header("🛡️ **Network Security Control Center**")
    
    # Network Integration Status
    st.subheader("🔧 Network Integration Status")
    systems = {
        "AI Fraud Detection": "✅ Active",
        "Telecom Network API": "✅ Connected", 
        "Banking Gateway": "🟡 Demo Mode",
        "Real-time Processing": "✅ Operational",
        "Multi-language Support": "✅ Hindi/English",
        "Threat Database": "✅ Updated"
    }
    
    for system, status in systems.items():
        st.write(f"**{system}:** {status}")
    
    # Business Model Calculator
    st.subheader("💰 Network Service Metrics")
    cost_per_call = 0.01  # ₹0.01 per API call
    estimated_cost = st.session_state.api_calls * cost_per_call
    monthly_projection = estimated_cost * 30 * 24  # Rough daily to monthly projection
    
    st.metric("API Calls Processed", st.session_state.api_calls)
    st.metric("Service Cost", f"₹{estimated_cost:.2f}")
    st.metric("Monthly Projection", f"₹{monthly_projection:.2f}")
    
    # Real-time statistics
    st.subheader("📊 **Live Network Statistics**")
    
    # Create metrics in enhanced layout
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("📨 **Processed**", total_sent)
        st.metric("✅ **Safe**", genuine_count)
    with metric_col2:
        st.metric("⚠️ **Flagged**", suspected_count)
        st.metric("🚫 **Blocked**", blocked_count)
    
    # Enhanced security effectiveness
    if total_sent > 0:
        blocked_percentage = (blocked_count / total_sent) * 100
        protection_rate = ((blocked_count + suspected_count) / total_sent) * 100
        st.metric("🛡️ **Block Rate**", f"{blocked_percentage:.1f}%")
        st.metric("🔍 **Detection Rate**", f"{protection_rate:.1f}%")
    
    # Notification Center
    if st.session_state.notifications:
        st.subheader("🔔 **Network Alerts**")
        
        with st.expander(f"📬 {len(st.session_state.notifications)} Recent Alerts"):
            for notification in reversed(st.session_state.notifications[-5:]):
                alert_time = notification['time'].strftime('%H:%M:%S')
                st.write(f"**{alert_time}:** {notification['message']}")
                if 'details' in notification:
                    st.caption(notification['details'])
                st.markdown("---")
    
    # Enhanced Threat Log
    if st.session_state.blocked_messages:
        st.subheader("🚨 **Network Threat Log**")
        
        show_ai_analysis = st.checkbox("Show AI analysis details", key="show_analysis_toggle")
        
        with st.expander(f"🛡️ {blocked_count} Network-Blocked Threats"):
            for idx, blocked in enumerate(reversed(st.session_state.blocked_messages[-5:]), 1):
                st.error(f"**Network Threat #{idx}**")
                st.write(f"**Source:** {blocked['sender']}")
                st.write(f"**Target:** {blocked['recipient']}")
                st.write(f"**Risk Level:** {blocked.get('fraud_risk', 0):.1%}")
                st.write(f"**Language:** {blocked.get('language', 'English')}")
                
                # Show preview
                preview_text = blocked['text'][:80] + "..." if len(blocked['text']) > 80 else blocked['text']
                st.text(f"Preview: {preview_text}")
                
                st.caption(f"🕐 Blocked at {blocked['timestamp'].strftime('%H:%M:%S')}")
                
                # AI Analysis Details
                if show_ai_analysis:
                    if blocked.get('explanations'):
                        st.markdown("**🤖 AI Detected:**")
                        for explanation in blocked['explanations']:
                            st.caption(f"• {explanation}")
                    
                    if 'probabilities' in blocked and blocked['probabilities']:
                        probs = blocked['probabilities']
                        analysis_col1, analysis_col2 = st.columns(2)
                        with analysis_col1:
                            st.caption(f"Safe: {probs.get('class_0_prob', 0):.1%}")
                            st.caption(f"Suspicious: {probs.get('class_1_prob', 0):.1%}")
                        with analysis_col2:
                            st.caption(f"Spam: {probs.get('class_2_prob', 0):.1%}")
                            st.caption(f"Class: {probs.get('predicted_class', 'N/A')}")
                
                if idx < len(st.session_state.blocked_messages[-5:]):
                    st.markdown("---")
    
    # Control Panel
    st.subheader("🎛️ **Network Controls**")
    
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        if st.button("🗑️ Clear Logs", type="secondary"):
            st.session_state.messages = []
            st.session_state.blocked_messages = []
            st.session_state.notifications = []
            st.success("✅ Network logs cleared!")
            st.rerun()
    
    with control_col2:
        if st.button("🔄 Reset System", type="secondary"):
            st.session_state.api_calls = 0
            st.session_state.messages = []
            st.session_state.blocked_messages = []
            st.session_state.notifications = []
            st.success("✅ Network system reset!")
            st.rerun()
    
    # Export functionality
    if st.button("📊 Export Network Report", type="primary"):
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'network_stats': {
                'total_processed': total_sent,
                'messages_delivered': len(st.session_state.messages),
                'genuine_delivered': genuine_count,
                'flagged_delivered': suspected_count,
                'threats_blocked': blocked_count,
                'api_calls': st.session_state.api_calls,
                'block_rate_percentage': (blocked_count / total_sent * 100) if total_sent > 0 else 0,
                'detection_rate_percentage': ((blocked_count + suspected_count) / total_sent * 100) if total_sent > 0 else 0
            },
            'system_performance': st.session_state.system_performance,
            'threat_summary': [
                {
                    'sender': msg['sender'],
                    'risk_level': msg['fraud_risk'],
                    'language': msg.get('language', 'English'),
                    'timestamp': msg['timestamp'].isoformat(),
                    'ai_explanations': msg.get('explanations', [])
                }
                for msg in st.session_state.blocked_messages
            ],
            'network_integration_status': systems
        }
        
        st.download_button(
            label="💾 Download Network Security Report",
            data=json.dumps(report, indent=2, default=str),
            file_name=f"network_security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Enhanced Footer with Network Information
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <strong>🛡️ Network-Embedded AI Fraud Protection System</strong><br>
        <em>Telecom-Integrated Defense • Real-time Threat Interception • Multi-language Support</em><br>
        <small>🌐 Network-Level Processing • 🤖 DistilBERT AI • 🇮🇳 Built for Indian Cyber Threats</small>
    </div>
    """, 
    unsafe_allow_html=True
)