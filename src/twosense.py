"""Alternative names for each class, including folder and synset senses."""

TWOSENSE = {
    "birds": ["bird", "wild bird", "songbird", "owl", "toucan", "flamingo", "peacock",
              "penguin", "eagle", "parrot", "hummingbird", "sparrow", "house finch"],
    "ducks": ["duck", "mallard duck", "duckling", "waterfowl", "wild duck", "goose"],
    "seals": ["seal", "harbor seal", "sea lion", "fur seal", "elephant seal", "seal pup",
              "walrus", "dugong", "manatee"],
    "fishes": ["fish", "tropical fish", "koi fish", "shark", "ray", "eel", "goldfish"],
    "snakes": ["snake", "serpent", "python", "cobra", "viper", "rattlesnake", "green mamba"],
    "cakes": ["cake", "layer cake", "birthday cake", "cheesecake", "bundt cake", "cupcake",
              "sponge cake", "trifle"],
    "breads": ["bread", "loaf of bread", "baguette", "bread roll", "sliced bread",
               "sourdough bread", "hot dog"],
    "bottles": ["bottle", "glass bottle", "water bottle", "wine bottle", "plastic bottle",
                "beer bottle"],
    "handguns": ["handgun", "pistol", "revolver", "firearm", "holster"],
    "lipsticks": ["lipstick", "lip gloss", "tube of lipstick", "makeup lipstick"],
    "vases": ["vase", "flower vase", "ceramic vase", "urn", "porcelain vase"],
    "lions": ["lion", "lioness", "male lion with mane", "lion cub"],
    "cats": ["cat", "domestic cat", "kitten", "tabby cat", "house cat"],
    "dogs": ["dog", "puppy", "domestic dog", "pet dog"],
    "cows": ["cow", "cattle", "calf", "dairy cow", "bull"],
    "horses": ["horse", "pony", "foal", "stallion", "mare"],
    "elephants": ["elephant", "african elephant", "asian elephant", "baby elephant"],
    "chickens": ["chicken", "hen", "rooster", "chick", "poultry"],
    "butterfiles": ["butterfly", "monarch butterfly", "moth", "swallowtail butterfly"],
    "spiders": ["spider", "tarantula", "orb weaver spider", "spider on a web"],
}

def names_for(classes):
    """Return names by class."""
    missing = [c for c in classes if c not in TWOSENSE]
    if missing:
        raise KeyError(f"no two-sense list for {missing}")
    return {c: list(TWOSENSE[c]) for c in classes}

if __name__ == "__main__":
    n = sum(len(v) for v in TWOSENSE.values())
    print(f"{len(TWOSENSE)} folders, {n} sub-concepts, "
          f"{n / len(TWOSENSE):.1f} per class")
    for k, v in TWOSENSE.items():
        print(f"  {k:12s} {len(v):2d}  {', '.join(v)}")
