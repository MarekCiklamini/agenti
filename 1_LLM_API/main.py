import requests
import json

# Konfigurace LLM API (nahraďte skutečnými údaji)
LLM_API_URL = "https://example.com/llm/api"
API_KEY = "vase_api_klíč"

# Funkce pro výpočet (v tomto případě jednoduchý kalkulátor)
def vypocitej(vyraz):
    """
    Provede výpočet na zadaném výrazu.
    """
    try:
        vysledek = eval(vyraz)  # Používejte eval s opatrností!
        return str(vysledek)
    except (SyntaxError, NameError, TypeError) as e:
        return f"Chyba ve výpočtu: {e}"

# Funkce pro volání LLM API
def zavolej_llm(prompt, nastroje):
    """
    Volá LLM API s daným promptem a seznamem nástrojů.
    """
    hlavicky = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "nastroje": nastroje
    }

    try:
        response = requests.post(LLM_API_URL, headers=hlavicky, data=json.dumps(data))
        response.raise_for_status()  # Vyvolá výjimku pro špatné status kódy
        odpoved = response.json()
        return odpoved
    except requests.exceptions.RequestException as e:
        print(f"Chyba při volání LLM API: {e}")
        return None

# Hlavní část skriptu
if __name__ == "__main__":
    # Příklad promptu, který vyžaduje použití nástroje
    prompt = "Kolik je 2 + 2 * 3?"

    # Definice nástroje (v tomto případě funkce 'vypocitej')
    nastroje = {
        "vypocitej": {
            "popis": "Provede matematický výpočet.",
            "parametry": {
                "vyraz": "Matematický výraz k vyhodnocení."
            }
        }
    }

    # Volání LLM API
    odpoved = zavolej_llm(prompt, nastroje)

    if odpoved:
        print("Odpověď LLM:")
        print(odpoved["text"])  # Předpokládá se, že odpověď obsahuje 'text'
```
