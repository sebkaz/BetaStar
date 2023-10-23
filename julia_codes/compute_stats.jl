using DataFrames
using Graphs
using PyCall
using Random
using Statistics
using StatsBase
using CSV

# On error run:
# run(`$(PyCall.python) -m pip install python-igraph`)
ig = pyimport("igraph")
pyrandom = pyimport("random")

function load_abcdo(suffix::AbstractString)
    ename = "edge_" * suffix * ".dat"
    community = "com_" * suffix * ".dat"

    edgeset = [Tuple(parse.(Int, split(line))) for line in eachline(ename)]
    target = map(endswith("\t1"), eachline(community))
    return edgeset, target
end

function load_reddit()
    prefix = "reddit"
    out = "1.0 "
    isoutlier = readlines(prefix * ".outliers") .== out
    graph = SimpleGraph(length(isoutlier))
    for line in eachline(prefix * ".edgelist")
        a, b = parse.(Int, split(line))
        a != b && add_edge!(graph, a, b)
    end
    cc = connected_components(graph)[1]
    gr1 = graph[cc]
    edgeset = Tuple.(collect(edges(gr1)))
    target = isoutlier[cc]
    open("reddit.dat", "w") do io
        for e in edgeset
            println(io, e[1], "\t", e[2])
        end
    end
    return edgeset, [i > 9998 ? missing : target[i] for i in 1:length(target)]
end

function load_grid()
    edge_list = split.(readlines("gridkit_europe-highvoltage.edges"))
    vertex_ids = unique(reduce(vcat, edge_list))
    vertex_map = Dict(vertex_ids .=> 1:length(vertex_ids))
    gr = SimpleGraph(length(vertex_ids))
    foreach(((from, to),) -> add_edge!(gr, vertex_map[from], vertex_map[to]), edge_list)
    df = CSV.read("gridkit_europe-highvoltage.vertices", DataFrame)
    df.id = [get(vertex_map, string(v), missing) for v in df.v_id]
    dropmissing!(sort!(df, :id))
    @assert df.id == 1:length(vertices(gr))

    cc = connected_components(gr)[1]
    gr1 = gr[cc]
    edgeset = Tuple.(collect(edges(gr1)))
    target = df.typ[cc] .== "plant"

    open("grid.dat", "w") do io
        for e in edgeset
            println(io, e[1], "\t", e[2])
        end
    end

    return edgeset, target
end

function load_lastfm()
    edge_df = CSV.read("lastfm_asia_edges.csv", DataFrame)
    vertex_ids = unique(Matrix(edge_df))
    vertex_map = Dict(vertex_ids .=> 1:length(vertex_ids))
    gr = SimpleGraph(length(vertex_ids))
    foreach(((from, to),) -> add_edge!(gr, vertex_map[from], vertex_map[to]), zip(eachcol(edge_df)...))
    df = CSV.read("lastfm_asia_target.csv", DataFrame)
    df.true_id = [vertex_map[id] for id in df.id]
    sort!(df, :true_id)
    @assert df.true_id == 1:length(vertices(gr))

    edgeset = Tuple.(collect(edges(gr)))
    target = df.target .== 17

    open("lastfm.dat", "w") do io
        for e in edgeset
            println(io, e[1], "\t", e[2])
        end
    end

    return edgeset, target
end

function load_facebook()
    edge_df = CSV.read("musae_facebook_edges.csv", DataFrame)
    vertex_ids = unique(Matrix(edge_df))
    vertex_map = Dict(vertex_ids .=> 1:length(vertex_ids))
    gr = SimpleGraph(length(vertex_ids))
    foreach(((from, to),) -> from != to && add_edge!(gr, vertex_map[from], vertex_map[to]), zip(eachcol(edge_df)...))
    df = CSV.read("musae_facebook_target.csv", DataFrame)
    df.true_id = [vertex_map[id] for id in df.id]
    sort!(df, :true_id)
    @assert df.true_id == 1:length(vertices(gr))

    edgeset = Tuple.(collect(edges(gr)))
    target = df.page_type .== "politician"

    open("facebook.dat", "w") do io
        for e in edgeset
            println(io, e[1], "\t", e[2])
        end
    end

    return edgeset, target
end

function load_amazon()
    target = readlines("target_amazon.txt")
    graph = SimpleGraph(length(target))
    for line in eachline("edge_amazon.dat")
        a, b = parse.(Int, split(line))
        @assert a != b
        add_edge!(graph, a, b)
    end
    cc = connected_components(graph)[3]
    gr1 = graph[cc]
    edgeset = Tuple.(collect(edges(gr1)))
    mapping = Dict("missing" => missing, "true" => true, "false" => false)
    target = [mapping[v] for v in target[cc]]
    open("amazon.dat", "w") do io
        for e in edgeset
            println(io, e[1], "\t", e[2])
        end
    end
    return edgeset, target
end

# Note that the alogirthms below were not optimized for speed, but to keep their implementation simple

function participation_coeff(g::SimpleGraph, ind::Int, comm::AbstractVector{Int})
    degv = degree(g, ind)
    return sum((degvi / degv)^2 for degvi in values(countmap(comm[neighbors(g, ind)])))
end

function in_mod_deg_vec(g::SimpleGraph, comm::AbstractVector{Int})
    ekv = [count(nei -> comm[nei] == comm[ind], neighbors(g, ind)) for ind in 1:nv(g)]
    tmp_df = DataFrame(; ekv, comm)
    tmp_df.id = 1:nv(g)
    transform!(groupby(tmp_df, :comm), :ekv .=> (x -> (x .- mean(x)) ./ std(x)) => :in_mod_deg)
    @assert tmp_df.id == 1:nv(g)
    return tmp_df.in_mod_deg
end

function beta_star(g::SimpleGraph, ind::Int, comm::AbstractVector{Int})
    ek = count(nei -> comm[nei] == comm[ind], neighbors(g, ind))
    degak = degree(g, ind)
    volAi = sum(degree(g)[comm .== comm[ind]])
    volV = sum(degree(g))
    return ek/degak - (volAi-degak)/volV
end

function edge_contr(g::SimpleGraph, ind::Int, comm::AbstractVector{Int})
    ek = count(nei -> comm[nei] == comm[ind], neighbors(g, ind))
    degak = degree(g, ind)
    return ek/degak
end

function cada(g::SimpleGraph, ind::Int, comm::AbstractVector{Int})
    ek = maximum(values(countmap(comm[neighbors(g, ind)])))
    degak = degree(g, ind)
    return degak/ek
end

function q_dist(clusters)
    cs = sort!(unique(clusters))
    @assert cs[1] == 1
    @assert cs[end] == length(cs)

    cs .= 0
    for c in clusters
        cs[c] += 1
    end

    return cs / sum(cs)
end

# note that we work with degreess so p < eps() should be the same as p == 0
log_diff((p, q),) = p < eps() ? 0.0 : p * log(p / q)

kl(p, q) = (@assert length(p) == length(q); sum(log_diff, zip(p, q)))

l1(p, q) = (@assert length(p) == length(q); sum(x -> abs(x[1] - x[2]), zip(p, q)))

l2(p, q) = (@assert length(p) == length(q); sqrt(sum(x -> (x[1] - x[2])^2, zip(p, q))))

hd(p, q) = (@assert length(p) == length(q); sqrt(sum(x -> (sqrt(x[1]) - sqrt(x[2]))^2, zip(p, q))) / sqrt(2))

function distances_gr(g::SimpleGraph, ind::Int, comm::AbstractVector{Int}, q::Vector{Float64})
    p = zeros(Float64, length(q))
    for nei in neighbors(g, ind)
        p[comm[nei]] += 1.0
    end
    p ./= sum(p)
    return (kl=kl(p, q), l1=l1(p, q), l2=l2(p, q), hd=hd(p, q))
end

function distances2_gr(g::SimpleGraph, ind::Int, comm::AbstractVector{Int}, q::Vector{Float64})
    p = zeros(Float64, length(q))
    for nei in Iterators.flatten(neighbors(g, x) for x in neighbors(g, ind))
        p[comm[nei]] += 1.0
    end
    p ./= sum(p)
    return (kl2=kl(p, q), l12=l1(p, q), l22=l2(p, q), hd2=hd(p, q))
end

function avg_neighbor_degree(g::SimpleGraph, ind::Int)
    return mean(x -> degree(g, x), neighbors(g, ind))
end

function compute_statistics(edgeset, target, out_file_name)
    df = DataFrame(node_id=1:length(target), target=target)

    n = length(target)
    g = Graphs.SimpleGraph(n)
    for (a, b) in edgeset
        add_edge!(g, a, b)
    end
    @assert ne(g) == length(edgeset)

    ig_g = ig.Graph.TupleList([string.("n", Tuple(e)) for e in edgeset])
    best_leiden_obj = nothing
    best_modularity = -1.0
    pyrandom.seed(1234)
    for i in 1:1000 # run Leiden 1000 times and pick best outcome
        i % 50 == 0 && @info "leiden @ $i"
        leiden_obj = ig_g.community_leiden(objective_function="modularity", n_iterations=-1)
        if leiden_obj.modularity > best_modularity
            best_leiden_obj = leiden_obj
            best_modularity = leiden_obj.modularity
        end
    end
    clusters_leiden = best_leiden_obj.membership .+ 1
    node_labels = fill("error", n)
    for v in ig_g.vs()
        node_labels[v.index + 1] = v.attributes()["name"]
    end
    # membership_dict = Dict(node_labels .=> clusters_ecg)
    membership_dict = Dict(node_labels .=> clusters_leiden)
    clusters_g = [membership_dict[string("n", i)] for i in 1:n]

    df.participation = [participation_coeff(g, i, clusters_g) for i in 1:n]
    df.in_mod_deg = in_mod_deg_vec(g, clusters_g)
    df.beta_star = [beta_star(g, i, clusters_g) for i in 1:n]
    df.edge_contr = [edge_contr(g, i, clusters_g) for i in 1:n]
    df.cada = [cada(g, i, clusters_g) for i in 1:n]

    q_clusters_g = q_dist(clusters_g)
    g_distances = [distances_gr(g, i, clusters_g, q_clusters_g) for i in 1:nv(g)]
    df.kl = getproperty.(g_distances, :kl)
    df.l1 = getproperty.(g_distances, :l1)
    df.l2 = getproperty.(g_distances, :l2)
    df.hd = getproperty.(g_distances, :hd)
    g_distances2 = [distances2_gr(g, i, clusters_g, q_clusters_g) for i in 1:nv(g)]
    df.kl2 = getproperty.(g_distances2, :kl2)
    df.l12 = getproperty.(g_distances2, :l12)
    df.l22 = getproperty.(g_distances2, :l22)
    df.hd2 = getproperty.(g_distances2, :hd2)
    df.lcc = local_clustering_coefficient(g, 1:nv(g))
    df.bc = betweenness_centrality(g)
    df.dc = degree(g)
    df.ndc = [avg_neighbor_degree(g, i) for i in 1:nv(g)]
    df.cc = closeness_centrality(g)
    df.ec = eigenvector_centrality(g)
    df.eccen = eccentricity(g)
    df.core = core_number(g)

    CSV.write(out_file_name, df)
end

compute_statistics(load_abcdo("x3_o1000")..., "x3_o1000.csv")
compute_statistics(load_abcdo("x4_o1000")..., "x4_o1000.csv")
compute_statistics(load_abcdo("x5_o1000")..., "x5_o1000.csv")
compute_statistics(load_abcdo("x6_o1000")..., "x6_o1000.csv")
compute_statistics(load_abcdo("x7_o1000")..., "x7_o1000.csv")

compute_statistics(load_reddit()..., "reddit.csv")
compute_statistics(load_grid()..., "grid.csv")
compute_statistics(load_lastfm()..., "lastfm.csv")
compute_statistics(load_facebook()..., "facebook.csv")
compute_statistics(load_amazon()..., "amazon.csv")
