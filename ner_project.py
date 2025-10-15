import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

def install_requirements():
    """Install required packages if not already installed"""
    try:
        import spacy
        import pandas
        import matplotlib
    except ImportError:
        print("Installing required packages...")
        os.system("pip install spacy pandas matplotlib")
        os.system("python -m spacy download en_core_web_sm")
        print("✅ Installation complete")

def load_spacy_model():
    """Load spaCy model"""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except IOError:
        print("❌ spaCy model not found. Installing...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        return nlp

def get_sample_texts():
    """Return sample texts for analysis"""
    return [
        "Barack Obama was the 44th President of the United States and was born in Hawaii.",
        "Apple Inc. is a technology company based in Cupertino, California, founded by Steve Jobs.",
        "Sachin Tendulkar is a legendary cricket player from Mumbai, India.",
        "Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975.",
        "The Eiffel Tower is a famous landmark located in Paris, France.",
        "Google was started by Larry Page and Sergey Brin at Stanford University.",
        "Lionel Messi plays football for Paris Saint-Germain and the Argentina national team.",
        "Amazon.com was founded by Jeff Bezos in Seattle, Washington."
    ]

def extract_entities(nlp, text):
    """Extract entities from text"""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'description': spacy.explain(ent.label_)
        })
    return entities

def analyze_texts(nlp, texts):
    """Analyze all texts and extract entities"""
    all_results = []
    
    print(" Analyzing texts...")
    print("=" * 50)
    
    for i, text in enumerate(texts):
        print(f"\n--- Text {i+1} ---")
        print(f"Text: {text}")
        print("Entities:")
        
        entities = extract_entities(nlp, text)
        
        if entities:
            for entity in entities:
                print(f"  - {entity['text']} ({entity['label']})")
                all_results.append({
                    'text_id': i+1,
                    'original_text': text,
                    'entity_text': entity['text'],
                    'entity_type': entity['label'],
                    'description': entity['description']
                })
        else:
            print("  - No entities found")
    
    print(f"\nTotal entities extracted: {len(all_results)}")
    return all_results

def create_analysis_report(results):
    """Create analysis report"""
    if not results:
        print("❌ No results to analyze")
        return None, None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate statistics
    total_texts = df['text_id'].nunique()
    total_entities = len(df)
    unique_entities = df['entity_text'].nunique()
    unique_types = df['entity_type'].nunique()
    avg_entities_per_text = total_entities / total_texts
    
    # Count entity types
    entity_counts = df['entity_type'].value_counts()
    common_entities = df['entity_text'].value_counts().head(10)
    
    print("\n NER ANALYSIS SUMMARY")
    print("=" * 40)
    print(f"Total texts analyzed: {total_texts}")
    print(f"Total entities found: {total_entities}")
    print(f"Unique entities: {unique_entities}")
    print(f"Entity types found: {unique_types}")
    print(f"Average entities per text: {avg_entities_per_text:.1f}")
    
    print("\n Entity Type Breakdown:")
    for entity_type, count in entity_counts.items():
        percentage = (count / total_entities) * 100
        description = spacy.explain(entity_type)
        print(f"- {entity_type}: {count} ({percentage:.1f}%) - {description}")
    
    print("\n Most Common Entities:")
    for entity, count in common_entities.head(5).items():
        print(f"- '{entity}': mentioned {count} times")
    
    return df, entity_counts

def save_results(df):
    """Save results to files"""
    try:
        # Save CSV
        df.to_csv('ner_results.csv', index=False)
        print("✅ Results saved to 'ner_results.csv'")
        
        # Save summary report
        summary_report = f"""NAMED ENTITY RECOGNITION (NER) ANALYSIS REPORT
=============================================

Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
- Total texts analyzed: {df['text_id'].nunique()}
- Total entities extracted: {len(df)}
- Unique entities found: {df['entity_text'].nunique()}
- Different entity types: {df['entity_type'].nunique()}

ENTITY TYPE BREAKDOWN:
"""
        
        entity_counts = df['entity_type'].value_counts()
        for entity_type, count in entity_counts.items():
            percentage = (count / len(df)) * 100
            summary_report += f"- {entity_type}: {count} entities ({percentage:.1f}%)\n"
        
        with open('ner_summary_report.txt', 'w') as f:
            f.write(summary_report)
        
        print("✅ Summary report saved to 'ner_summary_report.txt'")
        
    except Exception as e:
        print(f"⚠️ Could not save files: {e}")

def test_custom_text(nlp):
    """Test NER with custom text"""
    print("\n TESTING ANOTHER TEST SAMPLE")
    print("=" * 30)
    
    # You can modify this text
    custom_text = "Rohit Sharma, a cricketing genius known for his effortless elegance, destructive batting, and record-breaking centuries. His leadership, poise, and consistency have solidified him as one of the game’s finest, inspiring millions with each powerful stroke and calculated innings."
    
    print(f"Text: {custom_text}")
    print("\nEntities found:")
    
    doc = nlp(custom_text)
    if doc.ents:
        for ent in doc.ents:
            print(f"- '{ent.text}' → {ent.label_} ({spacy.explain(ent.label_)})")
    else:
        print("No entities found. Try adding some names, places, or organizations!")

def main():
    
    # Step 1: Install requirements (if needed)
    install_requirements()
    
    # Step 2: Load spaCy model
    nlp = load_spacy_model()
    
    # Step 3: Get sample texts
    sample_texts = get_sample_texts()
    print(f"\n {len(sample_texts)} sample texts")
    
    # Step 4: Analyze texts
    results = analyze_texts(nlp, sample_texts)
    
    if results:
        # Step 5: Create analysis report
        df, entity_counts = create_analysis_report(results)
        
        # Step 6: Save results
        if df is not None:
            save_results(df)
        
        # Step 7: Test custom text
        test_custom_text(nlp)
        
        print("\n PROJECT COMPLETED SUCCESSFULLY!")
        print("Check the generated files:")
        print("- ner_results.csv")
        print("- ner_summary_report.txt")
    
    else:
        print("❌ No entities found in the texts")

if __name__ == "__main__":
    main()
