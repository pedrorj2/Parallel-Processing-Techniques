# File: code/2_foundational_concepts/operation-to-data.jl

# Console explanation
println("Operation-to-Data Ratio Analysis")
println("\nMatrix-Vector (MxV) Multiplication:")
println("Total Operations (Nops) = 2N²")
println("Total Data Elements (Ndata) = N² + 2N")
println("Explanation: For large N, N² dominates 2N, so N² + 2N ≈ N², making the ratio 2N² / N² ≈ 2.")
println("\nMxV Operation-to-Data Ratio = 2N² / (N² + 2N) ≈ 2 (for large N)")

println("\nMatrix-Matrix (MxM) Multiplication:")
println("Total Operations (Nops) = 2N³")
println("Total Data Elements (Ndata) = 3N²")
println("Explanation: The N² terms cancel out, leaving the ratio 2N³ / 3N² = (2/3)N, which grows linearly.")
println("\nMxM Operation-to-Data Ratio = 2N³ / 3N² = (2/3)N")