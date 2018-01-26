int main()
{
#if defined (__AVX512CD__) && defined (__AVX512F__)
    return 0;
#else
#error // AVX512 NOT SUPPORTED
#endif
}
