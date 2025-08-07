from classify_graph import graph

reviews = [
    "The food was too spicy and made me uncomfortable.",
    "The delivery was delayed by 2 hours.",
    "The staff was very rude at the billing counter.",
    "I found the price way too high for such poor quality.",
    "Dessert was very sweet but I enjoyed it."
]

for review in reviews:
    result = graph.invoke({'review': review})
    print(f"\nReview: {review}")
    print(f"-> Classified: {result['level1']} > {result['level2']}")