//
// Created by mateusz on 04.12.18.
//
#include "GraphSession.h"

GraphSession::GraphSession()
{
	graph = TF_NewGraph();
	options = TF_NewSessionOptions();
	session = run_with_status<TF_Session*>(std::bind(TF_NewSession, graph, options, std::placeholders::_1));
}

GraphSession::~GraphSession()
{
	run_with_status<void>(std::bind(TF_DeleteSession, session, std::placeholders::_1));
	TF_DeleteSessionOptions(options);
	TF_DeleteGraph(graph);
}
