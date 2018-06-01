int main()
{
#if defined (__AVX__) || defined (__AVX2__)
    return 0;
#else
#error // AVX NOT SUPPORTED
#endif
}
