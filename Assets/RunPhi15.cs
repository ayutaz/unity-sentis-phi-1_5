using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using System.IO;
using Newtonsoft.Json;
using System.Text;
using TMPro;

/*
 *              Phi 1.5 Inference Code
 *              =======================
 *  
 *  Put this script on the Main Camera
 *  
 *  In Assets/StreamingAssets put:
 *  
 *  phi15.sentis
 *  vocab.json
 *  merges.txt
 * 
 *  Install package com.unity.nuget.newtonsoft-json from packagemanger
 *  Install package com.unity.sentis
 * 
 */


public class RunPhi15 : MonoBehaviour
{
    const BackendType backend = BackendType.GPUCompute;
    //string outputString = "Question: \"What is the capital of France?\"\n Correct answer: \"";
    //string outputString = "The human asked, \"What is your favourite animal?\" so the wise man answered correctly,  \"";
    string outputString = "Once upon a time, there were three";
    //string outputString = "// Javascript function for prime numbers";

    // This is how many tokens you want. It can be adjusted.
    const int maxTokens = 100;

    [SerializeField] private TextMeshProUGUI outputText;

    [SerializeField] private TextMeshProUGUI inputText;

    //Make this smaller for more randomness
    const float predictability = 5;

    //Special tokens
    const int END_OF_TEXT = 50256;

    Ops ops;
    ITensorAllocator allocator;

    //Store the vocabulary
    string[] tokens;

    IWorker engine;

    int currentToken = 0;
    int[] outputTokens = new int[maxTokens];

    // Used for special character decoding
    int[] whiteSpaceCharacters = new int[256];
    int[] encodedCharacters = new int[256];

    bool runInference = false;


    //stop after this many tokens
    const int stopAfter = 200;

    int totalTokens = 0;

    string[] merges;
    Dictionary<string, int> vocab;

    void Start()
    {
        allocator = new TensorCachingAllocator();
        ops = WorkerFactory.CreateOps(backend, allocator);

        inputText.text = "Input: " + outputString;

        SetupWhiteSpaceShifts();

        LoadVocabulary();
       
        Model model = ModelLoader.Load(Application.streamingAssetsPath + "/phi15.sentis");
        engine = WorkerFactory.CreateWorker(backend, model);

        GO(outputString);
    }

    public void GO(string text)
    {
        outputString = text;
        DecodePrompt(outputString);
        runInference = true;
    }

    // Update is called once per frame
    void Update()
    {
        if (runInference)
        {
            RunInference();
        }
    }

    void RunInference()
    {
        using var tokensSoFar = new TensorInt(new TensorShape(1, maxTokens), outputTokens);

        engine.Execute(tokensSoFar);

        var tokensOut = engine.PeekOutput() as TensorFloat;

        using var row = ops.Slice(tokensOut, new[] { currentToken }, new[] { currentToken + 1 }, new[] { 1 }, new[] { 1 });
        using var rowB = ops.Mul(predictability, row);
        using var probs = ops.Softmax(rowB, 2);
        probs.MakeReadable();

        int ID = SelectRandomToken(probs.ToReadOnlyArray());

        if (currentToken >= maxTokens - 1)
        {
            for (int i = 0; i < maxTokens - 1; i++) outputTokens[i] = outputTokens[i + 1];
            currentToken--;
        }

        outputTokens[++currentToken] = ID;
        totalTokens++;

        if (ID == END_OF_TEXT || totalTokens >= stopAfter)
        {
            runInference = false;
        }
        else if (ID < 0 || ID >= tokens.Length)
        {
            // Really we should use the added_tokens.json for this
            outputString += " ";
        }
        else outputString += GetUnicodeText(tokens[ID]);

        Debug.Log(outputString);
        outputText.text = "Output: " + outputString + "...";
    }

    void DecodePrompt(string text)
    {
        var inputTokens = GetTokens(text);

        for(int i = 0; i < inputTokens.Count; i++)
        {
            outputTokens[i] = inputTokens[i];
        }
        currentToken = inputTokens.Count - 1;
    }
   

    void LoadVocabulary()
    {
        var jsonText = File.ReadAllText(Application.streamingAssetsPath + "/vocab.json");
        vocab = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonText);
        tokens = new string[vocab.Count];
        foreach (var item in vocab)
        {
            tokens[item.Value] = item.Key;
        }

        merges = File.ReadAllLines(Application.streamingAssetsPath + "/merges.txt");
    }


    int SelectRandomToken(float[] probs)
    {
        float p = UnityEngine.Random.Range(0, 1f);
        float t = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            t += probs[i];
            if (p < t)
            {
                return i;
            }
        }
        return probs.Length - 1;
    }

    // Translates encoded special characters to Unicode
    string GetUnicodeText(string text)
    {
        var bytes = Encoding.GetEncoding("ISO-8859-1").GetBytes(ShiftCharacterDown(text));
        return Encoding.UTF8.GetString(bytes);
    }
    string GetASCIIText(string newText)
    {
        var bytes = Encoding.UTF8.GetBytes(newText);
        return ShiftCharacterUp(Encoding.GetEncoding("ISO-8859-1").GetString(bytes));
    }

    string ShiftCharacterDown(string text)
    {
        string outText = "";
        foreach (char letter in text)
        {
            outText += ((int)letter <= 256) ? letter :
                (char)whiteSpaceCharacters[(int)(letter - 256)];
        }
        return outText;
    }

    string ShiftCharacterUp(string text)
    {
        string outText = "";
        foreach (char letter in text)
        {
            outText += (char)encodedCharacters[(int)letter];
        }
        return outText;
    }

    void SetupWhiteSpaceShifts()
    {
        for (int i = 0, n = 0; i < 256; i++)
        {
            encodedCharacters[i] = i;
            if (IsWhiteSpace((char)i))
            {
                encodedCharacters[i] = n + 256;
                whiteSpaceCharacters[n++] = i;
            }
        }
    }

    bool IsWhiteSpace(char c)
    {
        return !(('!' <= c && c <= '~') || ('�' <= c && c <= '�') || ('�' <= c && c <= '�'));
    }

    List<int> GetTokens(string text)
    {
        text = GetASCIIText(text);

        // Start with a list of single characters
        var inputTokens = new List<string>();
        foreach(var letter in text)
        {
            inputTokens.Add(letter.ToString());
        }

        ApplyMerges(inputTokens);

        //Find the ids of the words in the vocab
        var ids = new List<int>();
        foreach(var token in inputTokens)
        {
            if (vocab.TryGetValue(token, out int id))
            {
                ids.Add(id);
            }
        }

        return ids;
    }

    void ApplyMerges(List<string> inputTokens)
    {
        foreach(var merge in merges)
        {
            string[] pair = merge.Split(' ');
            int n = 0;
            while (n >= 0)
            {
                n = inputTokens.IndexOf(pair[0], n);
                if (n != -1 && n < inputTokens.Count - 1 && inputTokens[n + 1] == pair[1])
                {
                    inputTokens[n] += inputTokens[n + 1];
                    inputTokens.RemoveAt(n + 1);
                }
                if (n != -1) n++;
            }
        }
    }

    private void OnDestroy()
    {
        engine?.Dispose();
        ops?.Dispose();
        allocator?.Dispose();
    }
}
