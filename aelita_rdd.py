from pyspark import SparkContext, SparkConf
import re
import os

# Функции вынесены отдельно для корректной сериализации в Spark
def load_russian_stopwords():
    """Загрузка русских стоп-слов"""
    return {
        'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
        'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
        'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
        'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
        'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него',
        'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом',
        'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо',
        'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без',
        'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда',
        'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним',
        'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
        'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец',
        'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через',
        'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три',
        'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда',
        'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда',
        'конечно', 'всю', 'между', 'это', 'вот', 'вдруг', 'как', 'под', 'над',
        'при', 'после', 'перед', 'через', 'из', 'от', 'до', 'для', 'за', 'к', 'у'
    }

def clean_text(line, stop_words):
    """
    1. TEXT CLEANING: Remove stop words, punctuation, numbers, etc.
    """
    if not line.strip():
        return []
    
    # Convert to lowercase
    line = line.lower().strip()
    
    # Remove punctuation, numbers, and Latin characters
    line = re.sub(r'[^а-яё\s]', ' ', line)
    line = re.sub(r'\s+', ' ', line)
    
    # Split into words and filter
    words = line.split()
    cleaned_words = []
    
    for word in words:
        word = word.strip()
        # Keep only Russian words longer than 2 characters that are not stop words
        if len(word) > 2 and word not in stop_words and re.match(r'^[а-яё]+$', word):
            cleaned_words.append(word)
            
    return cleaned_words

def simple_russian_stemmer(word):
    """
    4. STEMMING: Basic Russian stemming
    """
    endings = [
        'ость', 'ать', 'ять', 'ить', 'еть', 'уть', 'ють', 'ат', 'ят', 'ит', 
        'ет', 'ут', 'ют', 'ый', 'ий', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие', 
        'ов', 'ев', 'ам', 'ям', 'ами', 'ями', 'ах', 'ях', 'ом', 'ем', 'у', 
        'ю', 'ем', 'им', 'шь', 'ть', 'ти', 'л', 'ла', 'ло', 'ли', 'н', 'на', 
        'но', 'ны', 'ть', 'ся', 'ей', 'ой'
    ]
    
    # Remove endings (longest first)
    for ending in sorted(endings, key=len, reverse=True):
        if word.endswith(ending) and len(word) > len(ending) + 2:
            return word[:-len(ending)]
    return word

class RussianTextAnalyzer:
    def __init__(self):
        # Spark configuration for standalone mode
        conf = SparkConf().setAppName("RussianTextAnalysis") \
                          .setMaster("local[*]") \
                          .set("spark.driver.memory", "2g") \
                          .set("spark.executor.memory", "2g")
        
        self.sc = SparkContext(conf=conf)
        self.stop_words = load_russian_stopwords()
        
        print("=" * 70)
        print("SPARK TEXT ANALYSIS - RUSSIAN LANGUAGE")
        print("=" * 70)
    
    def analyze_text(self, file_path):
        """Main analysis method covering all 5 requirements"""
        
        if not os.path.exists(file_path):
            print(f"ERROR: File {file_path} not found!")
            return
        
        # Read text file
        text_rdd = self.sc.textFile(file_path)
        print(f"File loaded: {file_path}")
        
        # Broadcast stop words for use in transformations
        stop_words_broadcast = self.sc.broadcast(self.stop_words)
        
        # ===========================================================================
        # 1. TEXT CLEANING
        # ===========================================================================
        print("\n" + "=" * 50)
        print("1. TEXT CLEANING")
        print("=" * 50)
        
        cleaned_words_rdd = text_rdd.flatMap(lambda line: clean_text(line, stop_words_broadcast.value))
        
        total_words = cleaned_words_rdd.count()
        print(f"Total words after cleaning: {total_words:,}")
        
        if total_words == 0:
            print("No words found after cleaning!")
            return
        
        # ===========================================================================
        # 2. WORDCOUNT
        # ===========================================================================
        print("\n" + "=" * 50)
        print("2. WORDCOUNT DEVELOPMENT")
        print("=" * 50)
        
        wordcount_rdd = cleaned_words_rdd.map(lambda word: (word, 1)) \
                                        .reduceByKey(lambda a, b: a + b)
        
        unique_words = wordcount_rdd.count()
        print(f"Unique words: {unique_words:,}")
        
        # ===========================================================================
        # 3. TOP-50 MOST AND LEAST COMMON WORDS
        # ===========================================================================
        print("\n" + "=" * 50)
        print("3. TOP-50 MOST COMMON WORDS")
        print("=" * 50)
        
        top_50 = wordcount_rdd.takeOrdered(50, key=lambda x: -x[1])
        
        print(f"{'Rank':<4} {'Word':<20} {'Count':<8} {'Percentage':<10}")
        print("-" * 50)
        for i, (word, count) in enumerate(top_50, 1):
            percentage = (count / total_words) * 100
            print(f"{i:<4} {word:<20} {count:<8} {percentage:.2f}%")
        
        print("\n" + "=" * 50)
        print("3. TOP-50 LEAST COMMON WORDS")
        print("=" * 50)
        
        # Get words that appear only once
        least_common_rdd = wordcount_rdd.filter(lambda x: x[1] == 1)
        least_common_count = least_common_rdd.count()
        
        print(f"Words appearing only once: {least_common_count:,}")
        
        if least_common_count > 0:
            bottom_50 = least_common_rdd.takeOrdered(50, key=lambda x: x[0])
            print(f"{'Rank':<4} {'Word':<20} {'Count':<8}")
            print("-" * 35)
            for i, (word, count) in enumerate(bottom_50, 1):
                print(f"{i:<4} {word:<20} {count:<8}")
        else:
            # If no words appear only once, take the 50 least frequent
            bottom_50 = wordcount_rdd.takeOrdered(50, key=lambda x: (x[1], x[0]))
            print(f"{'Rank':<4} {'Word':<20} {'Count':<8}")
            print("-" * 35)
            for i, (word, count) in enumerate(bottom_50, 1):
                print(f"{i:<4} {word:<20} {count:<8}")
        
        # ===========================================================================
        # 4. STEMMING
        # ===========================================================================
        print("\n" + "=" * 50)
        print("4. STEMMING APPLICATION")
        print("=" * 50)
        
        stemmed_words_rdd = cleaned_words_rdd.map(simple_russian_stemmer)
        stemmed_wordcount_rdd = stemmed_words_rdd.map(lambda word: (word, 1)) \
                                                .reduceByKey(lambda a, b: a + b)
        
        stemmed_unique_words = stemmed_wordcount_rdd.count()
        print(f"Unique words after stemming: {stemmed_unique_words:,}")
        
        reduction = ((unique_words - stemmed_unique_words) / unique_words) * 100
        print(f"Vocabulary reduction: {reduction:.1f}%")
        
        # ===========================================================================
        # 5. TOP-50 MOST AND LEAST COMMON WORDS AFTER STEMMING
        # ===========================================================================
        print("\n" + "=" * 50)
        print("5. TOP-50 MOST COMMON WORDS AFTER STEMMING")
        print("=" * 50)
        
        top_50_stemmed = stemmed_wordcount_rdd.takeOrdered(50, key=lambda x: -x[1])
        
        print(f"{'Rank':<4} {'Word':<20} {'Count':<8} {'Percentage':<10}")
        print("-" * 50)
        for i, (word, count) in enumerate(top_50_stemmed, 1):
            percentage = (count / total_words) * 100
            print(f"{i:<4} {word:<20} {count:<8} {percentage:.2f}%")
        
        print("\n" + "=" * 50)
        print("5. TOP-50 LEAST COMMON WORDS AFTER STEMMING")
        print("=" * 50)
        
        least_common_stemmed_rdd = stemmed_wordcount_rdd.filter(lambda x: x[1] == 1)
        least_common_stemmed_count = least_common_stemmed_rdd.count()
        
        print(f"Words appearing only once after stemming: {least_common_stemmed_count:,}")
        
        if least_common_stemmed_count > 0:
            bottom_50_stemmed = least_common_stemmed_rdd.takeOrdered(50, key=lambda x: x[0])
            print(f"{'Rank':<4} {'Word':<20} {'Count':<8}")
            print("-" * 35)
            for i, (word, count) in enumerate(bottom_50_stemmed, 1):
                print(f"{i:<4} {word:<20} {count:<8}")
        else:
            bottom_50_stemmed = stemmed_wordcount_rdd.takeOrdered(50, key=lambda x: (x[1], x[0]))
            print(f"{'Rank':<4} {'Word':<20} {'Count':<8}")
            print("-" * 35)
            for i, (word, count) in enumerate(bottom_50_stemmed, 1):
                print(f"{i:<4} {word:<20} {count:<8}")
        
        # ===========================================================================
        # SUMMARY
        # ===========================================================================
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        
        most_common_original = top_50[0] if top_50 else ("N/A", 0)
        most_common_stemmed = top_50_stemmed[0] if top_50_stemmed else ("N/A", 0)
        
        print(f"Total words processed: {total_words:,}")
        print(f"Unique words (original): {unique_words:,}")
        print(f"Unique words (after stemming): {stemmed_unique_words:,}")
        print(f"Most common word (original): '{most_common_original[0]}' ({most_common_original[1]} times)")
        print(f"Most common word (stemmed): '{most_common_stemmed[0]}' ({most_common_stemmed[1]} times)")
        print(f"Vocabulary reduction after stemming: {reduction:.1f}%")
        
        # Clean up
        stop_words_broadcast.unpersist()
        self.sc.stop()
        
        print("\nAnalysis completed successfully!")

def main():
    analyzer = RussianTextAnalyzer()
    try:
        analyzer.analyze_text('dataset.txt')
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()