namespace llm.cs;

public class ActivationTensors
{
    public (int,int) Encoded; // (B, T, C)
    public (int,int) Ln1; // (L, B, T, C)
    public (int,int) Ln1Mean; // (L, B, T)
    public (int,int) Ln1Rstd; // (L, B, T)
    public (int,int) Qkv; // (L, B, T, 3*C)
    public (int,int) Atty; // (L, B, T, C)
    public (int,int) Preatt; // (L, B, NH, T, T)
    public (int,int) Att; // (L, B, NH, T, T)
    public (int,int) AttProj; // (L, B, T, C)
    public (int,int) Residual2; // (L, B, T, C)
    public (int,int) Ln2; // (L, B, T, C)
    public (int,int) Ln2Mean; // (L, B, T)
    public (int,int) Ln2Rstd; // (L, B, T)
    public (int,int) Fch; // (L, B, T, 4*C)
    public (int,int) FchGelu; // (L, B, T, 4*C)
    public (int,int) FcProj; // (L, B, T, C)
    public (int,int) Residual3; // (L, B, T, C)
    public (int,int) Lnf; // (B, T, C)
    public (int,int) LnfMean; // (B, T)
    public (int,int) LnfRstd; // (B, T)
    public (int,int) Logits; // (B, T, V)
    public (int,int) Probs; // (B, T, V)
    public (int,int) Losses; // (B, T)
}
